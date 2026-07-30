#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /var/lib/h43/CAMPAIGN/restore.conf" >&2
  exit 2
fi
config=$1
if [[ ${config} != /var/lib/h43/*/restore.conf || ! -f ${config} ]]; then
  echo "unsafe or missing H43 restore configuration: ${config}" >&2
  exit 2
fi
config_mode=$(stat -c '%a' "${config}")
if [[ $(stat -c '%u' "${config}") -ne 0 ]] || (( (8#${config_mode} & 8#022) != 0 )); then
  echo "H43 restore configuration must be root-owned and not group/world-writable" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${config}"
: "${H43_CAMPAIGN:?}" "${H43_NONCE:?}" "${H43_RANK1_HOSTNAME:?}"
: "${H43_RANK1_CONTAINER_NAME:?}" "${H43_RANK1_CONTAINER_ID:?}"
: "${H43_RANK1_PROCESS_MARKER:?}" "${H43_RANK1_MARKER:?}"
: "${H43_MARKER_ADDRESS:?}" "${H43_MARKER_PORT:?}"
: "${H43_COMMAND_TIMEOUT:?}" "${H43_RANK1_READY_TIMEOUT:?}"
: "${H43_MARKER_READY_TIMEOUT:?}"
if [[ ! ${H43_CAMPAIGN} =~ ^[a-z0-9-]+$ || ! ${H43_NONCE} =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid H43 campaign or nonce" >&2
  exit 2
fi

exec 9>"/run/lock/h43-${H43_CAMPAIGN}-rank1.lock"
flock -x 9
if [[ $(hostname) != "${H43_RANK1_HOSTNAME}" ]]; then
  echo "rank-1 restore is on the wrong host" >&2
  exit 1
fi

docker_timeout() {
  timeout --signal=TERM --kill-after=5s "${H43_COMMAND_TIMEOUT}s" docker "$@"
}

observed_id=$(docker_timeout inspect --format '{{.Id}}' "${H43_RANK1_CONTAINER_NAME}")
if [[ ${observed_id} != "${H43_RANK1_CONTAINER_ID}" ]]; then
  echo "rank-1 accepted container identity changed" >&2
  exit 1
fi
state=$(docker_timeout inspect --format '{{.State.Status}}' "${observed_id}")
case ${state} in
  running) ;;
  created|exited)
    docker_timeout start "${observed_id}" >/dev/null
    ;;
  *)
    echo "rank-1 accepted container is not safely startable: ${state}" >&2
    exit 1
    ;;
esac

started_at=$(docker_timeout inspect --format '{{.State.StartedAt}}' "${observed_id}")
deadline=$((SECONDS + H43_RANK1_READY_TIMEOUT))
stable=0
while (( SECONDS < deadline )); do
  state=$(docker_timeout inspect --format '{{.State.Status}}' "${observed_id}")
  top=$(docker_timeout top "${observed_id}" -eo pid,args 2>/dev/null || true)
  logs=$(docker_timeout logs --since "${started_at}" "${observed_id}" 2>&1 || true)
  if [[ ${state} == running && ${top} == *"${H43_RANK1_PROCESS_MARKER}"* && ${logs} == *"${H43_RANK1_MARKER}"* ]]; then
    stable=$((stable + 1))
    if (( stable >= 3 )); then
      break
    fi
  else
    stable=0
  fi
  sleep 1
done
if (( stable < 3 )); then
  echo "rank-1 accepted server did not reach the frozen readiness predicate" >&2
  exit 1
fi

marker_root="/run/h43-markers/${H43_CAMPAIGN}"
install -d -m 0700 "${marker_root}/${H43_NONCE}"
marker="${H43_NONCE}|${H43_RANK1_CONTAINER_ID}|${started_at}"
printf '%s\n' "${marker}" >"${marker_root}/${H43_NONCE}/index.html.tmp"
chmod 0600 "${marker_root}/${H43_NONCE}/index.html.tmp"
mv "${marker_root}/${H43_NONCE}/index.html.tmp" "${marker_root}/${H43_NONCE}/index.html"

marker_unit="h43-marker-${H43_CAMPAIGN}"
if ! systemctl is-active --quiet "${marker_unit}.service"; then
  systemctl reset-failed "${marker_unit}.service" 2>/dev/null || true
  systemd-run --quiet --collect --unit "${marker_unit}" \
    --property=RuntimeMaxSec=2700 \
    /usr/bin/python3 -m http.server "${H43_MARKER_PORT}" \
      --bind "${H43_MARKER_ADDRESS}" --directory "${marker_root}/${H43_NONCE}"
fi
deadline=$((SECONDS + H43_MARKER_READY_TIMEOUT))
while (( SECONDS < deadline )); do
  if [[ $(curl -fsS --max-time 2 "http://${H43_MARKER_ADDRESS}:${H43_MARKER_PORT}/") == "${marker}" ]]; then
    logger -t h43 "${H43_CAMPAIGN}: exact accepted rank 1 is ready"
    exit 0
  fi
  sleep 1
done
echo "rank-1 readiness marker did not become available" >&2
exit 1

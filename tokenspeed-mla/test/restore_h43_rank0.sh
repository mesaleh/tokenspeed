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
: "${H43_CAMPAIGN:?}" "${H43_NONCE:?}" "${H43_RANK0_HOSTNAME:?}"
: "${H43_RANK0_CONTAINER_NAME:?}" "${H43_RANK0_CONTAINER_ID:?}"
: "${H43_RANK0_PROCESS_MARKER:?}" "${H43_RANK0_MARKER:?}"
: "${H43_RANK1_CONTAINER_ID:?}"
: "${H43_MARKER_ADDRESS:?}" "${H43_MARKER_PORT:?}"
: "${H43_HEALTH_URL:?}" "${H43_COMMAND_TIMEOUT:?}"
: "${H43_GPU_INDEX:?}" "${H43_GPU_IDLE_TIMEOUT:?}"
: "${H43_RANK1_READY_TIMEOUT:?}" "${H43_HTTP_HEALTH_TIMEOUT:?}"
if [[ ! ${H43_CAMPAIGN} =~ ^[a-z0-9-]+$ || ! ${H43_NONCE} =~ ^[0-9a-f]{64}$ || \
      ! ${H43_GPU_INDEX} =~ ^[0-9]+$ || ! ${H43_GPU_IDLE_TIMEOUT} =~ ^[0-9]+$ ]]; then
  echo "invalid H43 campaign or nonce" >&2
  exit 2
fi

exec 9>"/run/lock/h43-${H43_CAMPAIGN}-rank0.lock"
flock -x 9
if [[ $(hostname) != "${H43_RANK0_HOSTNAME}" ]]; then
  echo "rank-0 restore is on the wrong host" >&2
  exit 1
fi

docker_timeout() {
  timeout --signal=TERM --kill-after=5s "${H43_COMMAND_TIMEOUT}s" docker "$@"
}

if [[ -n ${H43_EXPERIMENT_CONTAINER_NAME:-} || -n ${H43_EXPERIMENT_CONTAINER_ID:-} ]]; then
  : "${H43_EXPERIMENT_CONTAINER_NAME:?}" "${H43_EXPERIMENT_CONTAINER_ID:?}"
  if experiment_id=$(docker_timeout inspect --format '{{.Id}}' "${H43_EXPERIMENT_CONTAINER_NAME}" 2>/dev/null); then
    if [[ ${experiment_id} != "${H43_EXPERIMENT_CONTAINER_ID}" ]]; then
      echo "experiment container name resolves to an unexpected identity" >&2
      exit 1
    fi
    experiment_state=$(docker_timeout inspect --format '{{.State.Status}}' "${experiment_id}")
    if [[ ${experiment_state} == running ]]; then
      docker_timeout stop --time 10 "${experiment_id}" >/dev/null
    elif [[ ${experiment_state} != exited && ${experiment_state} != created ]]; then
      echo "experiment container has an unsafe state: ${experiment_state}" >&2
      exit 1
    fi
  fi
fi

observed_id=$(docker_timeout inspect --format '{{.Id}}' "${H43_RANK0_CONTAINER_NAME}")
if [[ ${observed_id} != "${H43_RANK0_CONTAINER_ID}" ]]; then
  echo "rank-0 accepted container identity changed" >&2
  exit 1
fi
state=$(docker_timeout inspect --format '{{.State.Status}}' "${observed_id}")
case ${state} in
  running) rank0_already_running=1 ;;
  created|exited) rank0_already_running=0 ;;
  *)
    echo "rank-0 accepted container is not safely startable: ${state}" >&2
    exit 1
    ;;
esac

if (( ! rank0_already_running )); then
  idle_deadline=$((SECONDS + H43_GPU_IDLE_TIMEOUT))
  idle_stable=0
  while (( SECONDS < idle_deadline )); do
    rows=$(timeout --signal=TERM --kill-after=5s "${H43_COMMAND_TIMEOUT}s" \
      nvidia-smi --id="${H43_GPU_INDEX}" --query-compute-apps=pid \
        --format=csv,noheader,nounits)
    if [[ -z ${rows} ]]; then
      idle_stable=$((idle_stable + 1))
      if (( idle_stable >= 3 )); then
        break
      fi
    else
      idle_stable=0
    fi
    sleep 1
  done
  if (( idle_stable < 3 )); then
    logger -p user.warning -t h43 \
      "${H43_CAMPAIGN}: GPU remained busy; attempting exact rank-0 recovery"
  fi
fi

marker_url="http://${H43_MARKER_ADDRESS}:${H43_MARKER_PORT}/"
marker_prefix="${H43_NONCE}|${H43_RANK1_CONTAINER_ID}|"
deadline=$((SECONDS + H43_RANK1_READY_TIMEOUT))
while (( SECONDS < deadline )); do
  observed_marker=$(curl -fsS --max-time 2 "${marker_url}" 2>/dev/null || true)
  if [[ ${observed_marker} == "${marker_prefix}"* ]]; then
    break
  fi
  sleep 1
done
if [[ ${observed_marker:-} != "${marker_prefix}"* ]]; then
  echo "rank-0 restore did not observe the exact rank-1 readiness marker" >&2
  exit 1
fi

state=$(docker_timeout inspect --format '{{.State.Status}}' "${observed_id}")
case ${state} in
  running) ;;
  created|exited)
    docker_timeout start "${observed_id}" >/dev/null
    ;;
  *)
    echo "rank-0 accepted container is not safely startable: ${state}" >&2
    exit 1
    ;;
esac

started_at=$(docker_timeout inspect --format '{{.State.StartedAt}}' "${observed_id}")
deadline=$((SECONDS + H43_HTTP_HEALTH_TIMEOUT))
while (( SECONDS < deadline )); do
  state=$(docker_timeout inspect --format '{{.State.Status}}' "${observed_id}")
  top=$(docker_timeout top "${observed_id}" -eo pid,args 2>/dev/null || true)
  logs=$(docker_timeout logs --since "${started_at}" "${observed_id}" 2>&1 || true)
  if [[ ${state} == running && ${top} == *"${H43_RANK0_PROCESS_MARKER}"* && ${logs} == *"${H43_RANK0_MARKER}"* ]] && \
      curl -fsS --max-time 5 "${H43_HEALTH_URL}" >/dev/null; then
    logger -t h43 "${H43_CAMPAIGN}: exact accepted rank 0 and endpoint are ready"
    exit 0
  fi
  sleep 2
done
echo "rank-0 accepted endpoint did not become healthy" >&2
exit 1

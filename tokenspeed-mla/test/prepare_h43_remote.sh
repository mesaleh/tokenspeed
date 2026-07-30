#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CANDIDATE_COMMIT REFERENCE_COMMIT CAMPAIGN LOCAL_STATE_JSON" >&2
  exit 2
fi
candidate_commit_input=$1
reference_commit_input=$2
campaign=$3
local_state=$4
if [[ ! ${campaign} =~ ^[a-z0-9-]{1,40}$ || -e ${local_state} ]]; then
  echo "invalid campaign or state path already exists" >&2
  exit 2
fi

repo_root=$(git rev-parse --show-toplevel)
contract_rel=tokenspeed-mla/test/h43_codebook_ab_contract.json
contract=${repo_root}/${contract_rel}
base_image=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["accepted_service"]["rank0"]["image_id"])' "${contract}")
rank0_host=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["accepted_service"]["rank0"]["ssh_host"])' "${contract}")
experiment_hostname=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["machine"]["container_hostname"])' "${contract}")
health_url=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["accepted_service"]["health_url"])' "${contract}")
gpu_index=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["machine"]["gpu_index"])' "${contract}")
candidate_commit=$(git rev-parse --verify "${candidate_commit_input}^{commit}")
reference_commit=$(git rev-parse --verify "${reference_commit_input}^{commit}")
if [[ ${candidate_commit} != $(git rev-parse HEAD) ]]; then
  echo "candidate commit must be the checked-out HEAD" >&2
  exit 2
fi
if [[ -n $(git status --porcelain --untracked-files=all) ]]; then
  echo "H43 preparation requires a completely clean worktree" >&2
  exit 2
fi

remote_root="/var/lib/h43/${campaign}/preparation"
temporary_root=$(mktemp -d)
trap 'rm -rf -- "${temporary_root}"' EXIT
candidate_tar="h43-${campaign}-candidate.tar"
reference_tar="h43-${campaign}-reference.tar"
git archive --format=tar --output="${temporary_root}/${candidate_tar}" "${candidate_commit}"
git archive --format=tar --output="${temporary_root}/${reference_tar}" "${reference_commit}"

ssh -o BatchMode=yes "mesaleh@${rank0_host}" bash -s -- \
  "${remote_root}" "${health_url}" <<'REMOTE_INIT'
set -euo pipefail
remote_root=$1
health_url=$2
if sudo -n test -e "${remote_root}"; then
  echo "remote H43 preparation root already exists" >&2
  exit 2
fi
curl -fsS --max-time 10 "${health_url}" >/dev/null
sudo -n install -d -m 0755 "${remote_root}"
sudo -n install -d -m 0755 \
  "${remote_root}/candidate/source" "${remote_root}/reference/source"
sudo -n install -d -m 0700 "${remote_root}/incoming"
REMOTE_INIT

for role in candidate reference; do
  archive=${candidate_tar}
  if [[ ${role} == reference ]]; then
    archive=${reference_tar}
  fi
  ssh -o BatchMode=yes "mesaleh@${rank0_host}" \
    "sudo -n tee '${remote_root}/incoming/${role}.tar' >/dev/null" \
    <"${temporary_root}/${archive}"
  ssh -o BatchMode=yes "mesaleh@${rank0_host}" \
    "sudo -n chmod 0600 '${remote_root}/incoming/${role}.tar'"
done

ssh -o BatchMode=yes "mesaleh@${rank0_host}" bash -s -- \
  "${remote_root}" "${campaign}" "${candidate_commit}" "${reference_commit}" \
  "${base_image}" "${experiment_hostname}" "${health_url}" "${gpu_index}" <<'REMOTE'
set -euo pipefail
remote_root=$1
campaign=$2
candidate_commit=$3
reference_commit=$4
base_image=$5
experiment_hostname=$6
health_url=$7
gpu_index=$8
curl -fsS --max-time 10 "${health_url}" >/dev/null
for role in candidate reference; do
  archive="${remote_root}/incoming/${role}.tar"
  if [[ $(sudo -n stat -c '%u %a' "${archive}") != "0 600" ]]; then
    echo "H43 source archive is not root-owned mode 0600: ${archive}" >&2
    exit 2
  fi
  sudo -n tar -xf "${archive}" -C "${remote_root}/${role}/source"
  sudo -n rm -f "${archive}"
done
sudo -n rmdir "${remote_root}/incoming"

for role in candidate reference; do
  commit=${candidate_commit}
  if [[ ${role} == reference ]]; then
    commit=${reference_commit}
  fi
  source_root="${remote_root}/${role}/source"
  sudo -n docker build \
    --build-arg "BASE_IMAGE=${base_image}" \
    --build-arg "H43_SOURCE_COMMIT=${commit}" \
    --build-arg "H43_SOURCE_ROLE=${role}" \
    --file "${remote_root}/candidate/source/tokenspeed-mla/test/Dockerfile.h43" \
    --tag "h43-${campaign}-${role}:${commit:0:12}" \
    "${source_root}" 2>&1 \
    | sudo -n tee "${remote_root}/${role}/build.log" >/dev/null
  sudo -n install -d -m 0755 "${remote_root}/${role}/work"
  sudo -n cp -a "${remote_root}/candidate/source/tokenspeed-mla/test/." \
    "${remote_root}/${role}/work/"
  image_id=$(sudo -n docker image inspect "h43-${campaign}-${role}:${commit:0:12}" --format '{{.Id}}')
  printf '%s\n' "${image_id}" | sudo -n tee "${remote_root}/${role}/work/H43_IMAGE_ID" >/dev/null
  printf '%s\n' "${commit}" | sudo -n tee "${remote_root}/${role}/work/H43_SOURCE_COMMIT" >/dev/null
  printf '%s\n' "${role}" | sudo -n tee "${remote_root}/${role}/work/H43_SOURCE_ROLE" >/dev/null
  sudo -n docker run --rm \
    --volume "${remote_root}/${role}/work:/work:ro" \
    "${image_id}" \
    python3 /work/build_h43_source_manifest.py --work /work \
    | sudo -n tee "${remote_root}/${role}/work/H43_D1_SOURCE_MANIFEST.sha256" >/dev/null
  manifest_digest=$(sudo -n sha256sum "${remote_root}/${role}/work/H43_D1_SOURCE_MANIFEST.sha256" | awk '{print $1}')
  cache_root="/var/lib/h43-codebook-cache/${manifest_digest}"
  sudo -n install -d -m 0755 "${cache_root}"
  installed_sha=$(sudo -n docker run --rm "${image_id}" python3 -c \
    'import hashlib,pathlib,tokenspeed_mla.mla_decode_fp8 as m; p=pathlib.Path(m.__file__); print(hashlib.sha256(p.read_bytes()).hexdigest())')
  for phase in cold warm; do
    runtime=(
      python3 /work/prebuild_h43_codebook_cache.py
      --contract /work/h43_codebook_ab_contract.json
      --cache-root "${cache_root}"
      --source-manifest-digest "${manifest_digest}"
      --installed-mla-sha256 "${installed_sha}"
      --phase "${phase}"
    )
    container_prefix=(
      sudo -n docker run --rm --gpus "\"device=${gpu_index}\""
      --hostname "${experiment_hostname}" \
      --env PYTHONDONTWRITEBYTECODE=1 \
      --env "CUTE_DSL_CACHE_DIR=${cache_root}" \
      --volume "${remote_root}/${role}/work:/work:ro" \
      --volume "${cache_root}:${cache_root}" \
    )
    if [[ ${phase} == cold ]]; then
      "${container_prefix[@]}" "${image_id}" "${runtime[@]}" \
        | sudo -n tee "${remote_root}/${role}/prebuild-${phase}.json" >/dev/null
    else
      "${container_prefix[@]}" --cap-add SYS_ADMIN \
        --volume "${remote_root}/${role}:/evidence" \
        "${image_id}" ncu --section LaunchStats --csv \
          --log-file /evidence/prebuild-warm-ncu.csv \
          "${runtime[@]}" \
        | sudo -n tee "${remote_root}/${role}/prebuild-${phase}.json" >/dev/null
      sudo -n python3 "${remote_root}/candidate/source/tokenspeed-mla/test/check_h43_no_kernel_launch.py" \
        --ncu-log "${remote_root}/${role}/prebuild-warm-ncu.csv" \
        | sudo -n tee "${remote_root}/${role}/prebuild-no-kernel.json" >/dev/null
    fi
    curl -fsS --max-time 10 "${health_url}" >/dev/null
  done
  cache_digest=$(sudo -n python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["cache_after"]["digest"])' \
    "${remote_root}/${role}/prebuild-warm.json")
  sudo -n python3 - "${remote_root}/${role}/identity.json" \
    "${role}" "${commit}" "${image_id}" "${manifest_digest}" \
    "${installed_sha}" "${cache_root}" "${cache_digest}" \
    "${remote_root}/${role}/prebuild-cold.json" \
    "${remote_root}/${role}/prebuild-warm.json" \
    "${remote_root}/${role}/prebuild-no-kernel.json" <<'PY'
import json,sys
import hashlib,pathlib
path,role,commit,image,manifest,installed,cache_root,cache_digest,cold,warm,no_kernel=sys.argv[1:]
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
with open(path,"w",encoding="utf-8") as handle:
    json.dump({"role":role,"commit":commit,"image_id":image,
               "source_manifest_digest":manifest,"installed_mla_sha256":installed,
               "cache_root":cache_root,"cache_artifact_digest":cache_digest,
               "prebuild_cold_sha256":sha(cold),"prebuild_warm_sha256":sha(warm),
               "prebuild_no_kernel_sha256":sha(no_kernel)},
              handle,sort_keys=True)
    handle.write("\n")
PY
done
curl -fsS --max-time 10 "${health_url}" >/dev/null
sudo -n python3 - "${remote_root}" "${campaign}" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1])
value={"schema_version":1,"status":"PREPARED","campaign":sys.argv[2]}
for role in ("candidate","reference"):
    value[role]=json.loads((root/role/"identity.json").read_text())
print(json.dumps(value,sort_keys=True))
PY
REMOTE

ssh -o BatchMode=yes "mesaleh@${rank0_host}" \
  "sudo -n cat '${remote_root}/candidate/identity.json'; sudo -n cat '${remote_root}/reference/identity.json'" \
  >"${temporary_root}/identities.txt"
python3 - "${temporary_root}/identities.txt" "${local_state}" "${campaign}" "${remote_root}" <<'PY'
import json,sys,pathlib
lines=pathlib.Path(sys.argv[1]).read_text().splitlines()
if len(lines)!=2:
    raise SystemExit("expected two remote identity records")
value={"schema_version":1,"status":"PREPARED","campaign":sys.argv[3],
       "remote_root":sys.argv[4],"candidate":json.loads(lines[0]),
       "reference":json.loads(lines[1])}
import hashlib
payload=json.dumps(value,sort_keys=True,separators=(",",":"))
value["state_digest"]=hashlib.sha256(payload.encode()).hexdigest()
pathlib.Path(sys.argv[2]).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
PY
python3 -m json.tool "${local_state}" >/dev/null
ssh -o BatchMode=yes "mesaleh@${rank0_host}" "curl -fsS --max-time 10 '${health_url}' >/dev/null"
echo "H43 remote preparation complete; accepted endpoint remained healthy"

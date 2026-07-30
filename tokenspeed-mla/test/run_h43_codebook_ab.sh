#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 qualification-smoke|qualification-preflight|decision-collect OUTPUT_ROOT DECISION_CONTRACT_OR_DASH" >&2
  exit 2
fi

mode=$1
output_root=$2
decision_contract=$3
work_root=/work
contract=${work_root}/h43_codebook_ab_contract.json
benchmark=${work_root}/microbench_h43_codebook_ab.py
source_manifest=${work_root}/H43_D1_SOURCE_MANIFEST.sha256

required_environment=(
  H43_SOURCE_MANIFEST_DIGEST
  H43_INSTALLED_MLA_SHA256
  H43_CACHE_ROOT
  H43_AOT_MANIFEST
  H43_AOT_EXPECTED_ENTRIES
  H43_AGGREGATE_ECC_BASELINE
  H43_EXPECTED_CACHE_DIGEST
)
for name in "${required_environment[@]}"; do
  if [[ -z ${!name:-} ]]; then
    echo "missing required environment: ${name}" >&2
    exit 2
  fi
done

case ${mode} in
  qualification-smoke|qualification-preflight|decision-collect) ;;
  *)
    echo "invalid collection mode: ${mode}" >&2
    exit 2
    ;;
esac

if [[ ! -f ${contract} || ! -f ${benchmark} || ! -f ${source_manifest} ]]; then
  echo "H43 staged contract, benchmark, or source manifest is missing" >&2
  exit 2
fi
if [[ ! -d ${H43_CACHE_ROOT} ]]; then
  echo "H43 compiled cache is missing: ${H43_CACHE_ROOT}" >&2
  exit 2
fi
mkdir -p "${output_root}"
source_log="${output_root}/source-verification-${mode}.log"
if [[ -e ${source_log} ]]; then
  echo "source-verification evidence already exists: ${source_log}" >&2
  exit 2
fi
sha256sum --check --strict "${source_manifest}" >"${source_log}"
observed_manifest_digest=$(sha256sum "${source_manifest}" | awk '{print $1}')
if [[ ${observed_manifest_digest} != "${H43_SOURCE_MANIFEST_DIGEST}" ]]; then
  echo "source-manifest digest mismatch" >&2
  exit 2
fi

export H43_PHYSICAL_HOST=$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["machine"]["physical_host"])' "${contract}"
)
export PYTHONDONTWRITEBYTECODE=1
export CUTE_DSL_CACHE_DIR=${H43_CACHE_ROOT}

mapfile -t contexts < <(
  python3 -c 'import json,sys; print(*json.load(open(sys.argv[1]))["contexts"], sep="\n")' "${contract}"
)
case_timeout=$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["timeouts_seconds"]["case_process"])' "${contract}"
)

run_case() {
  local case_dir=$1
  shift
  if [[ -e ${case_dir} ]]; then
    echo "H43 case directory already exists: ${case_dir}" >&2
    return 2
  fi
  mkdir -p "${case_dir}"
  set +e
  timeout --signal=TERM --kill-after=15s "${case_timeout}s" \
    python3 "${benchmark}" \
      --contract "${contract}" \
      --cache-root "${H43_CACHE_ROOT}" \
      --source-manifest-digest "${H43_SOURCE_MANIFEST_DIGEST}" \
      --installed-mla-sha256 "${H43_INSTALLED_MLA_SHA256}" \
      --aggregate-ecc-baseline "${H43_AGGREGATE_ECC_BASELINE}" \
      --expected-cache-digest "${H43_EXPECTED_CACHE_DIGEST}" \
      "$@" \
      >"${case_dir}/stdout.log" \
      2>"${case_dir}/stderr.log"
  local status=$?
  set -e
  printf '%s\n' "${status}" >"${case_dir}/exit.code"
  if [[ ${status} -ne 0 ]]; then
    return "${status}"
  fi
  tail -n 1 "${case_dir}/stdout.log" >"${case_dir}/result.json"
  python3 -m json.tool "${case_dir}/result.json" >/dev/null
}

if [[ ${mode} == qualification-smoke ]]; then
  for context in "${contexts[@]}"; do
    run_case "${output_root}/qualification/smoke/context${context}" \
      --mode smoke --context "${context}" --sequence 1
  done
elif [[ ${mode} == qualification-preflight ]]; then
  for sequence in 1 2; do
    for context in "${contexts[@]}"; do
      run_case "${output_root}/qualification/preflight/context${context}/seq0${sequence}" \
        --mode preflight --context "${context}" --sequence "${sequence}"
    done
  done
else
  if [[ ! -f ${decision_contract} ]]; then
    echo "decision contract is missing: ${decision_contract}" >&2
    exit 2
  fi
  process_count=$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["processes_per_context"])' \
      "${decision_contract}"
  )
  allowed_counts=$(
    python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["pilot"]; print(f"{p['"'"'base_processes_per_context'"'"']} {p['"'"'expanded_processes_per_context'"'"']}")' "${contract}"
  )
  if [[ " ${allowed_counts} " != *" ${process_count} "* ]]; then
    echo "invalid decision process count: ${process_count}" >&2
    exit 2
  fi
  for context in "${contexts[@]}"; do
    for sequence in $(seq 1 "${process_count}"); do
      run_case "${output_root}/decision/context${context}/seq$(printf '%02d' "${sequence}")" \
        --mode decision --context "${context}" --sequence "${sequence}"
    done
  done
fi

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[a-z0-9-]+$ ]]; then
  echo "usage: $0 CAMPAIGN" >&2
  exit 2
fi
message="H43 maintenance campaign $1 exceeded its alert threshold; verify accepted CT13+CT14 service restoration immediately"
logger -p user.alert -t h43 "${message}"
if command -v wall >/dev/null 2>&1; then
  printf '%s\n' "${message}" | wall || true
fi

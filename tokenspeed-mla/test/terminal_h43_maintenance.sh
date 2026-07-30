#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[a-z0-9-]+$ ]]; then
  echo "usage: $0 CAMPAIGN" >&2
  exit 2
fi
campaign=$1
for timer in \
  "h43-restore-${campaign}.timer" \
  "h43-alert-${campaign}.timer" \
  "h43-terminal-${campaign}.timer"; do
  systemctl stop "${timer}" 2>/dev/null || true
done
for service in \
  "h43-restore-${campaign}.service" \
  "h43-alert-${campaign}.service"; do
  systemctl reset-failed "${service}" 2>/dev/null || true
done
message="${campaign}: H43 terminal bound reached; automatic actions stopped, manual escalation required"
logger -p user.err -t h43 "${message}"
wall "${message}" 2>/dev/null || true

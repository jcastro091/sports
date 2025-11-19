#!/usr/bin/env bash
set -euo pipefail

HB_FILE="${HEARTBEAT_FILE_AUTOPOSTER:-/var/run/ss_autoposter_heartbeat}"
STALE_SEC="${STALE_SEC:-180}"
SERVICE_NAME="${SERVICE_NAME:-autoposter}"

if [[ ! -f "$HB_FILE" ]]; then
  echo "[watchdog] no heartbeat file, restarting $SERVICE_NAME"
  systemctl restart "$SERVICE_NAME".service || true
  python3 /opt/sports-repo/ops/notify_tg.py "♻️ $SERVICE_NAME restarted (no heartbeat file)"
  exit 0
fi

age=$(( $(date +%s) - $(stat -c %Y "$HB_FILE") ))
if (( age > STALE_SEC )); then
  echo "[watchdog] stale heartbeat ($age s), restarting $SERVICE_NAME"
  systemctl restart "$SERVICE_NAME".service || true
  python3 /opt/sports-repo/ops/notify_tg.py "♻️ $SERVICE_NAME restarted (stale heartbeat: ${age}s)"
fi

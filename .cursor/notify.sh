#!/usr/bin/env bash
# Fail-open ntfy ping of STATUS.md for iOS / remote-control. No secrets.
set +e
cat >/dev/null 2>&1 || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATUS="$ROOT/docs/architecture/STATUS.md"
TOPIC="${NTFY_TOPIC:-kinuv_canfar_agent_thbrown}"

if [[ ! -f "$STATUS" ]]; then
  printf '%s\n' '{}'
  exit 0
fi

BODY="$(python3 - "$STATUS" <<'PY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text(errors="replace").splitlines()
lines = []
grab = False
for line in text:
    if line.strip() == "## Agent Run Status":
        grab = True
        lines.append(line)
        continue
    if grab:
        if line.startswith("## ") and line.strip() != "## Agent Run Status":
            break
        lines.append(line)
        if len(lines) >= 12:
            break
if not lines:
    lines = text[:12]
print("\n".join(lines)[:900])
PY
)"

if command -v curl >/dev/null 2>&1; then
  curl -sS --max-time 8 \
    -H "Title: kinUV STATUS" \
    -H "Tags: computer" \
    -d "$BODY" \
    "https://ntfy.sh/${TOPIC}" >/dev/null 2>&1 || true
fi

printf '%s\n' '{}'
exit 0

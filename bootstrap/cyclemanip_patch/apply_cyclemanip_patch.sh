```bash
#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around apply_cyclemanip_patch.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine RoboTwin target directory.
if [[ ${1:-} == -* || ${1:-} == --* || -z ${1:-} ]]; then
  ROBOTTWIN_DIR="$(pwd)"
  EXTRA_ARGS=("$@")
else
  ROBOTTWIN_DIR="$1"
  shift
  EXTRA_ARGS=("$@")
fi

python3 "$SCRIPT_DIR/apply_cyclemanip_patch.py" --robottwin "$ROBOTTWIN_DIR" "${EXTRA_ARGS[@]}"

```

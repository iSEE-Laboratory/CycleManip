#!/usr/bin/env bash
# Prepare bbhl_demogen/loop1 folder structure (match bbhl_demogen/loop1-5).
set -euo pipefail

ROBOTWIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${ROBOTWIN_ROOT}/data/utils/prepare_bbhl_demogen_loop_dataset.py" \
  --setting loop1 \
  --n 100 \
  --loop_value 1 \
  --scene_info_src "${ROBOTWIN_ROOT}/data/beat_block_hammer_loop/loop1-8-counter_blhl/scene_info.json" \
  --scene_info_src_key episode_0 \
  --language_num 20

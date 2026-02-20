#!/bin/bash
# 用于处理真实机器人数据(只有头部相机,只有右臂)

data_dir=${1}
repo_id=${2}

echo "Converting real robot data to LeRobot format..."
echo "Data directory: $data_dir"
echo "Repository ID: $repo_id"
echo "Configuration: Single head camera, right arm only"

uv run examples/aloha_real/convert_aloha_data_to_lerobot_real.py --raw_dir $data_dir --repo_id $repo_id

echo "✓ Conversion completed!"

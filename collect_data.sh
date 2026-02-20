#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config

# sh collect_data.sh beat_block_hammer demo_randomized 0

# sh collect_data.sh shake_bottle_loop binary_

# sh collect_data.sh shake_bottle_loop loop
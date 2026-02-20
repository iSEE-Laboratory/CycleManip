#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data_loop.py $task_name $task_config

# sh collect_data_loop.sh shake_bottle_loop demo_loop_clean 0

# sh collect_data_loop.sh shake_bottle_loop loop
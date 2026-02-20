#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}

python scripts/process_data_loop.py $task_name $task_config $expert_data_num

# sh process_data.sh shake_bottle_loop binary_
# sh process_data.sh shake_bottle_loop loop

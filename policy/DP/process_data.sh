#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}

python process_data_debug.py $task_name $task_config $expert_data_num
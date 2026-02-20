#!/bin/bash

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source .venv/bin/activate
cd ../.. # move to root

# pip install piper_sdk
# pip install pyrealsense2==2.54.1.5216

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_realworld.py \
    --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --robot_ip can_right \
    --use_instruction True \
    --dont_stop True

# 敲三次 lora
# bash eval_real.sh beat_block_hammer_loop_real loop3 pi0_base_aloha_real_lora beat_block_hammer_loop_real_loop3 0 0

# 敲三次 full
# bash eval_real.sh beat_block_hammer_loop_real loop3 pi0_base_aloha_real_full beat_block_hammer_loop_real_loop3 0 0

# 敲1-8次 lora
# bash eval_real.sh beat_block_hammer_loop_real loop1-8 pi0_base_aloha_real_lora beat_block_hammer_loop_real_loop1-8 0 0

# 敲1-8次 full
# bash eval_real.sh beat_block_hammer_loop_real loop1-8 pi0_base_aloha_real_full beat_block_hammer_loop_real_loop1-8 0 0
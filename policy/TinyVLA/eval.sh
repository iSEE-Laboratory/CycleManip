#!/bin/bash
#
#policy_name=TinyVLAv2
#task_name=${1}
#task_config=${2}
#ckpt_setting=${3}
#seed=${4}
# gpu_id=${5}

policy_name=TinyVLAv2
task_name=place_object_scale
task_config=0
ckpt_setting=0
seed=0
gpu_id=0
# [TODO] add parameters here

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}
    --eval_video_log True
    # [TODO] add parameters here

#!/bin/bash

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
data_path=${7}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source .venv/bin/activate
cd ../.. # move to root

echo "python script/eval_policy_virtual.py --config policy/$policy_name/deploy_policy.yml --data_path $data_path --overrides --task_name ${task_name} --task_config ${task_config} --train_config_name ${train_config_name} --model_name ${model_name} --seed ${seed} --policy_name ${policy_name} --dont_stop True"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_virtual.py \
    --config policy/$policy_name/deploy_policy.yml \
    --data_path $data_path \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --use_instruction True \
    --dont_stop True

# c /home/scc/cuixj/workspace2/code/LoopBreaker/data/beat_block_hammer_loop_real/loop3/data/episode0.hdf5 


# bash eval_real_virtual.sh shake_bottle_loop_real loop3 pi0_base_aloha_real_lora shake_bottle_loop_real_loop3 0 0 /home/dex/haoran/gello_software/data_processed/shake_bottle_loop_real/loop3/data/episode0.hdf5
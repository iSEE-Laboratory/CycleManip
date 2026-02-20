#!/bin/bash

# == keep unchanged ==
policy_name=DP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --dont_stop True 
# cd ../DP
# bash eval.sh beat_block_hammer_loop loop1-8-all loop1-8-all 200 0 4
# bash eval.sh grab_roller_loop loop1-8-all loop1-8-all 200 0 5
# bash eval.sh shake_bottle_loop loop1-8-all loop1-8-all 200 0 6
# bash eval.sh double_knife_chop loop1-8-all loop1-8-all 200 0 7
# bash eval.sh cut_carrot_knife loop1-8-all loop1-8-all 200 0 4
# bash eval.sh beat_egg_loop loop1-8-all loop1-8-all 200 0 5
# bash eval.sh shake_flask_dropper_loop loop1-8-all-sfdl loop1-8-all-sfdl 200 0 6
# bash eval.sh morse_sos loop1-8-all loop1-8-all 200 0 7



# bash train.sh shake_bottle_loop loop1-8-all 200 0 14 3
# bash train.sh double_knife_chop loop1-8-all 200 0 14 4
# bash train.sh cut_carrot_knife loop1-8-all 200 0 14 6

# bash train.sh beat_egg_loop loop1-8-all 200 0 14 5
# bash train.sh shake_flask_dropper_loop loop1-8-all-sfdl 200 0 14 6
# bash train.sh morse_sos loop1-8-all 200 0 14 7




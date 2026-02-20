#!/bin/bash
# 
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}

head_camera_type=D435

DEBUG=False
save_ckpt=True

alg_name=robot_dp_$action_dim
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

python train.py --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            setting=${task_config} \
                            expert_data_num=${expert_data_num} \
                            head_camera_type=$head_camera_type
                            # checkpoint.save_ckpt=${save_ckpt}
                            # hydra.run.dir=${run_dir} \

# bash train.sh beat_block_hammer_loop loop1-8-all 200 0 14 0
# bash train.sh grab_roller_loop loop1-8-all 200 0 14 1
# bash train.sh shake_bottle_loop loop1-8-all 200 0 14 3
# bash train.sh double_knife_chop loop1-8-all 200 0 14 4
# bash train.sh cut_carrot_knife loop1-8-all 200 0 14 6

# bash train.sh beat_egg_loop loop1-8-all 200 0 14 5
# bash train.sh shake_flask_dropper_loop loop1-8-all-sfdl 200 0 14 6
# bash train.sh morse_sos loop1-8-all 200 0 14 7


# cd ../DP
# bash eval.sh beat_block_hammer_loop loop1-8-all loop1-8-all 200 0 4
# cd ../DP
# bash eval.sh grab_roller_loop loop1-8-all loop1-8-all 200 0 5
# cd ../DP
# bash eval.sh shake_bottle_loop loop1-8-all loop1-8-all 200 0 6
# cd ../DP
# bash eval.sh double_knife_chop loop1-8-all loop1-8-all 200 0 7
# cd ../DP
# bash eval.sh cut_carrot_knife loop1-8-all loop1-8-all 200 0 4
# cd ../DP
# bash eval.sh beat_egg_loop loop1-8-all loop1-8-all 200 0 5
# cd ../DP
# bash eval.sh shake_flask_dropper_loop loop1-8-all loop1-8-all 200 0 6
# cd ../DP
# bash eval.sh morse_sos loop1-8-all loop1-8-all 200 0 7
DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2} # task choices: See TASK.md
setting=${3}
expert_data_num=${4}
exp_tag=${5}
seed=${6}
gpu_id=${7}

config_name=${alg_name}
exp_name=${task_name}-${setting}-${exp_tag}
run_dir="experiments/${task_name}/${exp_name}/seed_${seed}"

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

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python dp3_train.py --config-name=${config_name}.yaml \
                            task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            expert_data_num=${expert_data_num} \
                            setting=${setting} \
                            exp_tag=${exp_tag}
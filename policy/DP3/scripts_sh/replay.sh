#!/bin/bash

policy_name=DP3
config_name=${1}
task_name=${2}
task_config=${3}
expert_data_num=${4}
exp_tag=${5}
seed=${6} 
gpu_id=${7}

exp_name=${task_name}-${task_config}-${exp_tag}


dir=experiments/${task_name}/${task_name}-${task_config}-${exp_tag}/seed_${seed}/wo_color_pc

# 初始值（无则保留为空）
max_num=-1
max_file=""

# 使用 find + -print0 安全处理带空格/特殊字符的文件名
# 这里 maxdepth=1 表示仅当前目录；如果你想递归，把 -maxdepth 1 去掉
while IFS= read -r -d '' f; do
    base=$(basename "$f")          # 例如 "1500.ckpt" 或 "model-1500.ckpt"
    name="${base%.ckpt}"           # 去掉后缀 -> "1500" 或 "model-1500"
    # 提取名称末尾的连续数字（如果有）
    if [[ $name =~ ([0-9]+)$ ]]; then
        num=${BASH_REMATCH[1]}
        # 比较并保留最大值
        if (( num > max_num )); then
            max_num=$num
            max_file="$f"1----+++++++++++
        fi
    fi
done < <(find "$dir" -maxdepth 1 -type f -name "*.ckpt" -print0 2>/dev/null)

if (( max_num >= 0 )); then
    echo "$max_num"
    # 如果你也想同时输出对应文件路径，取消下一行注释
    # echo "对应文件：$max_file"
else
    echo "未找到ckpt或未检测到文件名末尾的数字" >&2
    exit 1
fi


export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \

config_path=policy/$policy_name/3D-Diffusion-Policy/diffusion_policy_3d/deploy_config/deploy_policy_$config_name.yml

if [ -f "$config_path" ]; then
    echo "文件存在：$config_path"
else
    echo "文件不存在：$config_path"
    exit 1
fi

python script/replay_data.py --config $config_path \
    --overrides \
    --task_name ${task_name} \
    --exp_name ${exp_name} \
    --task_config ${task_config} \
    --ckpt_setting ${task_config} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --exp_tag ${exp_tag} \
    --checkpoint_num $max_num \
    --dont_stop True 

# bash eval.sh shake_bottle_loop loop1-8 200 instruction_sim sim+int_frozen_countertask 0 1
# bash eval.sh beat_block_hammer_loop loop1-8 200 instruction_sim sim+int_frozen_countertask 0 1
# bash eval.sh double_knife_chop loop1-8 200 instruction_sim sim+int_frozen_countertask 0 5
# bash eval.sh cut_carrot_knife loop1-8 200 instruction_sim sim+int_frozen_countertask 0 5

# bash scripts_sh/replay.sh binary_robot_dp3_obs6_ori beat_block_hammer_loop loop1-8-counter 200 ori_dp3 0 6

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
min_num=-1
max_file=""
min_file=""

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
            max_file="$f"
        fi
        
        # 比较并保留最小值
        if (( min_num == -1 || num < min_num )); then
            min_num=$num
            min_file="$f"
        fi
    fi
done < <(find "$dir" -maxdepth 1 -type f -name "*.ckpt" -print0 2>/dev/null)

if (( max_num >= 0 )); then
    echo "最大值: $max_num"
    # 如果你也想同时输出对应文件路径，取消下一行注释
    # echo "对应文件：$max_file"
else
    echo "未找到ckpt或未检测到文件名末尾的数字" >&2
    exit 1
fi

if (( min_num >= 0 )); then
    echo "最小值: $min_num"
    # 如果你也想同时输出对应文件路径，取消下一行注释
    # echo "对应文件：$min_file"
else
    echo "未找到最小值" >&2
    exit 1
fi


export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \

config_path=policy/$policy_name/3D-Diffusion-Policy/diffusion_policy_3d/deploy_config/deploy_policy_$config_name.yml

if [ -f "$config_path" ]; then
    echo "✅ 配置文件已存在：$config_path"
else
    echo "⚠️ 配置文件不存在，正在自动生成：$config_path"

    python policy/$policy_name/scripts/create_deploy_config.py \
        --policy_name "$policy_name" \
        --config_name "$config_name"

    # 再次检查是否生成成功
    if [ ! -f "$config_path" ]; then
        echo "❌ 配置文件生成失败，请检查 create_deploy_config.py"
        exit 1
    fi

    echo "✅ 配置文件已成功生成：$config_path"
fi

python script/eval_policy.py --config $config_path \
    --overrides \
    --task_name ${task_name} \
    --exp_name ${exp_name} \
    --task_config ${task_config} \
    --ckpt_setting ${task_config} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --exp_tag ${exp_tag} \
    --checkpoint_num $min_num \
    --dont_stop True 

# bash eval.sh shake_bottle_loop loop1-8 200 instruction_sim sim+int_frozen_countertask 0 1
# bash eval.sh beat_block_hammer_loop loop1-8 200 instruction_sim sim+int_frozen_countertask 0 1
# bash eval.sh double_knife_chop loop1-8 200 instruction_sim sim+int_frozen_countertask 0 5
# bash eval.sh cut_carrot_knife loop1-8 200 instruction_sim sim+int_frozen_countertask 0 5


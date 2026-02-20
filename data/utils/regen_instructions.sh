#!/bin/bash

# 只重新生成instruction，不重新采集数据
# 用法: bash regen_instructions.sh <task_name> [loop_config]
# 例如: bash regen_instructions.sh beat_block_hammer_loop loop8
# 或者: bash regen_instructions.sh beat_block_hammer_loop (自动处理所有loop配置)

cd ../..

task_name=${1}
specific_config=${2}

# 定义所有可能的loop配置
all_loop_configs=("loop1" "loop2" "loop3" "loop4" "loop5" "loop6" "loop7" "loop8" "loop1-4" "loop1-8" "loop5-8" "loop1-8-no-4" "loop1-8-no-6")

process_config() {
    local task_name=$1
    local task_config=$2
    
    # 检查数据目录是否存在
    data_dir="./data/${task_name}/${task_config}"
    if [ ! -d "$data_dir" ]; then
        echo "⏭️  跳过 ${task_config}: 数据目录不存在"
        return 1
    fi
    
    # 从配置文件中读取language_num
    config_file="./task_config/${task_config}.yml"
    if [ ! -f "$config_file" ]; then
        echo "⚠️  跳过 ${task_config}: 配置文件不存在"
        return 1
    fi
    
    # 使用grep和awk提取language_num的值
    language_num=$(grep "^language_num:" "$config_file" | awk '{print $2}')
    
    if [ -z "$language_num" ]; then
        echo "警告: 无法从配置文件中读取language_num，使用默认值100"
        language_num=10
    fi
    
    echo ""
    echo "=========================================="
    echo "任务名称: ${task_name}"
    echo "配置名称: ${task_config}"
    echo "指令数量: ${language_num}"
    echo "数据目录: ${data_dir}"
    echo "=========================================="
    
    cd description
    bash gen_episode_instructions.sh "${task_name}" "${task_config}" "${language_num}"
    cd ..
    
    echo "✅ ${task_config} 指令重新生成完成!"
    return 0
}

if [ -z "$task_name" ]; then
    echo "错误: 缺少任务名称参数"
    echo "用法: bash regen_instructions.sh <task_name> [loop_config]"
    echo "例如: bash regen_instructions.sh beat_block_hammer_loop loop8"
    echo "或者: bash regen_instructions.sh beat_block_hammer_loop (处理所有存在的loop配置)"
    exit 1
fi

# 如果指定了具体的配置，只处理该配置
if [ -n "$specific_config" ]; then
    process_config "$task_name" "$specific_config"
    exit $?
fi

# 否则，遍历所有可能的loop配置
echo "🔍 开始检测所有可能的loop配置..."
processed_count=0
skipped_count=0

for loop_config in "${all_loop_configs[@]}"; do
    if process_config "$task_name" "$loop_config"; then
        ((processed_count++))
    else
        ((skipped_count++))
    fi
done

echo ""
echo "======================================"
echo "📊 处理完成统计:"
echo "   ✅ 成功处理: ${processed_count} 个配置"
echo "   ⏭️  跳过: ${skipped_count} 个配置"
echo "======================================"

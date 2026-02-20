#!/bin/bash

# 定义目标目录（请替换为实际目录路径）
dir="/home/liaohaoran/code/RoboTwin/policy/DP3/experiments/double_knife_chop/double_knife_chop-loop1-8-counter-sim_frozen_timestask/seed_0/wo_color_pc/"

# 查找所有 ckpt 文件并提取数字部分，输出最大值
max_num=$(ls "$dir"/*.ckpt 2>/dev/null | \
    grep -oE '[0-9]+' | \
    sort -n | \
    tail -n 1)

if [ -n "$max_num" ]; then
    echo "$max_num"
else
    echo "未找到数字"
fi

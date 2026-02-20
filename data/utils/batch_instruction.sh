task_name=${1}
task_cfg=${2}


echo "开始生成仅包含简单指令的指令文件"

# 检查是否有instructions文件夹，没有就
if [ ! -d "../${task_name}/${task_cfg}/instructions" ]; then
    echo "instructions文件夹不存在，创建中..."
    bash regen_instructions.sh $task_name $task_cfg
fi

python fix_loop_time.py $task_name $task_cfg

echo "替换[num]"

mv ../${task_name}/${task_cfg}/instructions ../${task_name}/${task_cfg}/instructions_full

echo "开始生成仅包含数字的指令文件"

python gen_only_num_instruction.py $task_name $task_cfg

mv ../${task_name}/${task_cfg}/instructions ../${task_name}/${task_cfg}/instructions_int

# 请手动换instruction，按回车继续
read -p "手动换instruction, 完事了敲回车" key

echo "重新生成完整指令文件"

bash regen_instructions.sh $task_name $task_cfg

echo "替换[num]"

python fix_loop_time.py $task_name $task_cfg
mv ../${task_name}/${task_cfg}/instructions ../${task_name}/${task_cfg}/instructions_sim

# 给定任务名，config名
# 读取对应的data文件夹下的数据
# 读取loop_time.txt
# 解析
# 追加到原来的每一条数据的instruction的末尾
# 保存
import os
import json
import random
import argparse


def fix_loop_time(task_name, config_name):
    """
    为指定任务的instruction添加循环次数信息
    
    Args:
        task_name: 任务名称，如 'beat_block_hammer_loop'
        config_name: 配置名称，如 'loop8'
    """
    # 多样化的循环次数表达模板
    loop_templates = [
        "for {} times",
        "loop {} times",
        "repeat {} times",
        "{} times",
        "do it {} times",
        "perform {} times",
        "{} iterations",
        "execute {} times"
    ]
    
    # 构建数据路径
    base_path = os.path.join(os.path.dirname(__file__), '..', task_name, config_name)
    loop_times_file = os.path.join(base_path, 'loop_times.txt')
    instructions_dir = os.path.join(base_path, 'instructions')
    
    # 检查路径是否存在
    if not os.path.exists(loop_times_file):
        print(f"错误: 找不到文件 {loop_times_file}")
        return
    
    if not os.path.exists(instructions_dir):
        print(f"错误: 找不到目录 {instructions_dir}")
        return
    
    # 读取loop_times.txt
    with open(loop_times_file, 'r') as f:
        content = f.read().strip()
        loop_times_list = [int(x) for x in content.split()]
    
    print(f"读取到 {len(loop_times_list)} 个循环次数")
    
    # 遍历所有episode文件
    # 每一个文件是episode{n}.json的形式
    # 需要按照n，用数字排序，从小到大
    episode_files = [f for f in os.listdir(instructions_dir) if f.startswith('episode') and f.endswith('.json')]
    episode_files.sort(key=lambda x: int(x.replace('episode', '').replace('.json', '')))
    
    if len(episode_files) != len(loop_times_list):
        print(f"警告: episode文件数量 ({len(episode_files)}) 与循环次数数量 ({len(loop_times_list)}) 不匹配")
    
    for idx, episode_file in enumerate(episode_files):
        if idx >= len(loop_times_list):
            print(f"警告: {episode_file} 没有对应的循环次数，跳过")
            continue
        
        episode_path = os.path.join(instructions_dir, episode_file)
        loop_time = loop_times_list[idx]
        
        # 读取JSON文件
        with open(episode_path, 'r') as f:
            data = json.load(f)
        
        # 随机选择一个模板
        template = random.choice(loop_templates)
        suffix = f", {template.format(loop_time)}"
        
        # 为每个类别的instruction追加循环次数信息
        modified = False
        for category in ['seen', 'unseen']:
            if category in data:
                # 更新每条instruction
                for i in range(len(data[category])):
                    instruction = data[category][i]
                    
                    # 如果有[num]占位符，直接替换
                    if '[num]' in instruction:
                        data[category][i] = instruction.replace('[num]', str(loop_time))
                        modified = True
                    # 否则，检查是否需要添加循环次数信息
                    elif not any(keyword in instruction.lower() for keyword in ['times', 'loop', 'repeat', 'iterations', 'execute', 'perform']):
                        data[category][i] += suffix
                        modified = True
        
        # 保存修改后的文件
        if modified:
            with open(episode_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"已更新 {episode_file}: 添加 '{suffix}'")
        else:
            print(f"跳过 {episode_file}: 已包含循环次数信息")
    
    print(f"\n完成! 处理了 {len(episode_files)} 个episode文件")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='为指定任务的instruction添加循环次数信息')
    parser.add_argument('task_name', help='任务名称，例如: cut_carrot_knife')
    parser.add_argument('config_name', help='配置名称，例如: loop8')
    args = parser.parse_args()

    fix_loop_time(args.task_name, args.config_name)

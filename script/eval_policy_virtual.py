"""
Virtual Real World Policy Evaluation Script
用于在虚拟环境中测试真机部署 pipeline
使用 HDF5 数据文件模拟真机执行流程
"""
import sys
import os
sys.path.append("./")
sys.path.append("./policy")

import numpy as np
import yaml
import argparse
import importlib
from pathlib import Path
from datetime import datetime
import time
import json
import random
from termcolor import cprint

# 导入虚拟环境

# 单臂夹爪 -- 摇瓶子，锤锤子
# from envs.realworld.piper_virtual_env import PiperVirtualEnv as VirtualEnv

# 双臂Revo2手 -- 单刀切
from envs.realworld.piper_revo2_virtual_env import PiperRevo2VirtualEnv as VirtualEnv


def eval_function_decorator(policy_name, model_name):
    """动态加载 policy 函数（与仿真保持一致）"""
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def load_instruction_template(task_name):
    """从 instruction_template 文件夹加载指令模板"""
    template_path = Path(__file__).parent.parent / "envs" / "realworld" / "instruction_template" / f"{task_name}.json"
    
    if not template_path.exists():
        cprint(f"⚠️  警告: 指令模板未找到 {template_path}", "yellow")
        return None
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f)
    
    cprint(f"✓ 成功加载指令模板: {template_path}", "green")
    return template


def generate_instruction_from_template(template, loop_time):
    """根据模板和循环次数生成指令（同时生成 instruction_int 和 instruction_sim）
    
    Args:
        template: 指令模板字典
        loop_time: 循环次数
    
    Returns:
        instruction_dict: 包含 instruction_int 和 instruction_sim 的字典
    """
    # instruction_int: 只是数字
    instruction_int = str(loop_time)
    
    # instruction_sim: 从模板生成
    if template is None:
        # 如果没有模板，使用默认格式
        instruction_sim = f"perform the task {loop_time} times"
    else:
        # 随机选择 seen 或 unseen
        use_seen = random.choice([True, False])
        verb_list = template.get("seen" if use_seen else "unseen", {}).get("verb", [])
        state_list = template.get("seen" if use_seen else "unseen", {}).get("state", [])
        
        if verb_list and state_list:
            verb = random.choice(verb_list)
            state = random.choice(state_list).replace("[num]", str(loop_time))
            instruction_sim = f"{verb} {state}"
        else:
            instruction_sim = f"perform the task {loop_time} times"
    
    return {
        "instruction_int": instruction_int,
        "instruction_sim": instruction_sim
    }


def eval_policy_virtual(
    policy_name: str,
    model,
    virtual_env: VirtualEnv,
    args: dict,
    loop_times_array: list,
    instruction_template: dict,
):
    """在虚拟环境中评估策略
    
    Args:
        policy_name: policy 名称 (如 'ACT', 'DP3')
        model: 加载好的模型
        virtual_env: 虚拟环境
        args: 配置参数
        loop_times_array: 循环次数数组
        instruction_template: 指令模板字典
    """
    cprint("\n" + "=" * 60, "cyan", attrs=["bold"])
    cprint(f"开始虚拟环境评估（测试 Pipeline）", "cyan", attrs=["bold"])
    cprint(f"Policy: {policy_name}", "yellow")
    cprint(f"数据文件: {virtual_env.data_path.name}", "yellow")
    cprint(f"数据长度: {virtual_env.data_length} 帧", "yellow")
    cprint(f"循环次数数组: {loop_times_array}", "yellow")
    cprint("=" * 60 + "\n", "cyan", attrs=["bold"])
    
    # 动态加载 policy 的 eval 和 reset 函数
    eval_func = eval_function_decorator(policy_name, "eval_real")
    reset_func = eval_function_decorator(policy_name, "reset_model_real")
    
    cprint("🔄 重置环境和模型...", "yellow")
    virtual_env.reset()
    reset_func(model)
    cprint("✅ 重置完成\n", "green")
    
    # 自动生成指令（根据循环次数数组和模板）
    this_loop_time = args.get('loop_time', loop_times_array[0] if loop_times_array else 1)
    cprint(f"🔄 当前循环次数: {this_loop_time}", "yellow")
    
    # 使用虚拟环境的指令生成方法（从模板生成，同时包含 instruction_int 和 instruction_sim）
    instruction_dict = generate_instruction_from_template(
        instruction_template, 
        this_loop_time
    )
    
    cprint(f"📝 instruction_int: {instruction_dict['instruction_int']}", "blue")
    cprint(f"📝 instruction_sim: {instruction_dict['instruction_sim']}", "blue")
    virtual_env.set_instruction(
        instruction=instruction_dict["instruction_sim"],  # 这里没有full, 直接用sim
        instruction_int=instruction_dict["instruction_int"], 
        instruction_sim=instruction_dict["instruction_sim"]
    )
    
    if args.get('wait_for_start', False):
        input("\n⏸️  按 Enter 开始执行策略...\n")
    
    cprint("\n🚀 开始执行策略...\n", "green", attrs=["bold"])
    start_time = time.time()
    
    # 执行策略主循环
    step_count = 0
    max_steps = min(virtual_env.data_length, virtual_env.step_lim)
    
    try:
        while virtual_env.take_action_cnt < max_steps:
            # 获取观测
            observation = virtual_env.get_obs()
            
            cprint(f"\n{'='*60}", "blue")
            cprint(f"Step {virtual_env.take_action_cnt + 1}/{max_steps}", "blue", attrs=["bold"])
            cprint(f"{'='*60}", "blue")
            
            # 执行策略（会自动调用 take_action）
            cprint(f"\n🎯 执行策略推理...", "magenta")
            actions = eval_func(virtual_env, model, observation)
            
            step_count += 1
            
            # 控制播放速度
            if args.get('step_delay', 0) > 0:
                time.sleep(args['step_delay'])
            
            # 检查是否需要暂停
            if args.get('pause_each_step', False):
                cont = input("\n继续下一步? (Enter/q-退出): ").strip().lower()
                if cont == 'q':
                    cprint("用户中断", "red")
                    break
            
        elapsed_time = time.time() - start_time
        cprint(f"\n\n{'='*60}", "green", attrs=["bold"])
        cprint(f"⏱️  总执行时间: {elapsed_time:.2f}秒", "green")
        cprint(f"📊 总步数: {step_count}", "green")
        cprint(f"⚡ 平均步速: {step_count/elapsed_time:.2f} steps/sec", "green")
        cprint(f"{'='*60}\n", "green", attrs=["bold"])
        
    except KeyboardInterrupt:
        cprint("\n\n⚠️  用户中断执行", "red")
    except Exception as e:
        cprint(f"\n\n❌ 执行出错: {e}", "red", attrs=["bold"])
        import traceback
        traceback.print_exc()
    
    cprint("\n✅ 虚拟环境测试完成!", "green", attrs=["bold"])


def main(usr_args):
    """主函数"""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    exp_tag = usr_args["exp_tag"]
    data_path = usr_args['data_path']
    
    # 生成循环次数数组
    loop_times_min = usr_args.get('loop_times_min', 1)
    loop_times_max = usr_args.get('loop_times_max', 8)
    loop_times_array = list(range(loop_times_min, loop_times_max + 1))
    
    # 加载指令模板
    instruction_template = load_instruction_template(task_name)
    
    cprint("\n" + "=" * 60, "cyan", attrs=["bold"])
    cprint("虚拟真机环境 - Pipeline 测试", "cyan", attrs=["bold"])
    cprint("=" * 60, "cyan", attrs=["bold"])
    cprint(f"Policy: {policy_name}", "yellow")
    cprint(f"Task: {task_name}", "yellow")
    cprint(f"Task Config: {task_config}", "yellow")
    cprint(f"Checkpoint Setting: {ckpt_setting}", "yellow")
    cprint(f"Exp Tag: {exp_tag}", "yellow")
    cprint(f"Checkpoint Num: {checkpoint_num}", "yellow")
    cprint(f"数据文件: {data_path}", "yellow")
    cprint(f"Loop Times Array: {loop_times_array}", "cyan")
    cprint(f"Instruction Template: {'✓ Loaded' if instruction_template else '✗ Not Found (using default)'}", "yellow")
    cprint(f"Model_name: {usr_args.get('model_name', 'N/A')}", "yellow")
    cprint("=" * 60 + "\n", "cyan", attrs=["bold"])
    
    # 1. 加载模型
    cprint("📦 加载模型...", "cyan")
    get_model = eval_function_decorator(policy_name, "get_model_real")
    model = get_model(usr_args)
    cprint("✅ 模型加载完成\n", "green")
    
    # 2. 初始化虚拟环境
    cprint("🔧 初始化虚拟环境...", "cyan")
    virtual_env = VirtualEnv(
        policy=policy_name,
        data_path=data_path,
        step_lim=usr_args.get('step_lim', 500),
        verbose=usr_args.get('verbose', True)
    )
    
    # 3. 执行评估
    eval_policy_virtual(
        policy_name=policy_name,
        model=model,
        virtual_env=virtual_env,
        args=usr_args,
        loop_times_array=loop_times_array,
        instruction_template=instruction_template
    )
    
    # 4. 清理
    virtual_env.close()


def parse_args_and_config():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='虚拟真机环境 - Pipeline 测试'
    )
    parser.add_argument('--config', type=str, required=True, 
                        help='配置文件路径 (YAML)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='HDF5 数据文件路径')
    parser.add_argument('--dont_stop', type=str, choices=['true', 'false'], default='false', 
                        help="Disable early stopping on success (true/false)")
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细信息')
    parser.add_argument('--pause_each_step', action='store_true',
                        help='每步暂停等待确认')
    parser.add_argument('--step_delay', type=float, default=0.0,
                        help='每步延迟时间（秒）')
    parser.add_argument('--loop_time', type=int, default=None,
                        help='指定循环次数（如不指定则使用 loop_times_min）')
    parser.add_argument('--overrides', nargs=argparse.REMAINDER,
                        help='覆盖配置参数')
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 添加命令行参数
    config['data_path'] = args.data_path
    config['verbose'] = args.verbose
    config['pause_each_step'] = args.pause_each_step
    config['step_delay'] = args.step_delay
    if args.loop_time is not None:
        config['loop_time'] = args.loop_time
    
    # Add dont_stop to config (convert string to boolean)
    config["dont_stop"] = args.dont_stop.lower() == 'true'
    
    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip('--')
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict
    
    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)
    
    return config


if __name__ == "__main__":
    print("let's go!!!")
    cprint("\n" + "=" * 60, "cyan", attrs=["bold"])
    cprint("🧪 Virtual Real World Policy Evaluation", "cyan", attrs=["bold"])
    cprint("   (Pipeline Testing with HDF5 Data)", "cyan")
    cprint("=" * 60 + "\n", "cyan", attrs=["bold"])
    
    try:
        usr_args = parse_args_and_config()
        main(usr_args)
    except KeyboardInterrupt:
        cprint("\n\n⚠️  程序被用户中断", "red")
    except Exception as e:
        cprint(f"\n\n❌ 程序出错: {e}", "red", attrs=["bold"])
        import traceback
        traceback.print_exc()

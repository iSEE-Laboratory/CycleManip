"""
Piper Real World Policy Evaluation Script
用于在 Piper 真机上评估训练好的策略
"""
import sys
import os

from termcolor import cprint
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

# 导入真机环境

# 单臂夹爪 -- 摇瓶子，锤锤子
# from envs.realworld.piper_real_env import PiperRealEnv as RealEnv

# 双臂夹爪 -- 打鼓
# from envs.realworld.piper_real_env_bi import PiperRealEnv as RealEnv

# 双臂Revo2手 -- 单刀切
from envs.realworld.piper_revo2_real_env_bi import PiperRealEnv as RealEnv

# 人形机器人
# from envs.realworld.humanoid_real_env import HumanoidRealEnv as RealEnv

from envs.realworld.camera import get_device_ids


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


def eval_policy_realworld(
    policy_name: str,
    model,
    real_env: RealEnv,
    args: dict,
    loop_times_array: list,
    instruction_template: dict,
    test_num: int = 10,
):
    """在真机上评估策略
    
    Args:
        policy_name: policy 名称 (如 'ACT', 'DP3')
        model: 加载好的模型
        real_env: Piper 真机环境
        args: 配置参数
        loop_times_array: 循环次数数组
        instruction_template: 指令模板字典
        test_num: 测试轮数
    """
    cprint("\n" + "=" * 60, "cyan")
    cprint("🚀 开始真机评估", "cyan", attrs=["bold"])
    cprint(f"Policy: {policy_name}", "cyan")
    cprint(f"测试轮数: {test_num}", "cyan")
    cprint(f"循环次数数组: {loop_times_array}", "cyan")
    cprint("=" * 60 + "\n", "cyan")
    
    # 动态加载 policy 的 eval 和 reset 函数
    eval_func = eval_function_decorator(policy_name, "eval_real")
    reset_func = eval_function_decorator(policy_name, "reset_model_real")
    
    success_count = 0
    task_success_episodes = []
    task_failure_episodes = []
    loop_success_episodes = []
    loop_failure_episodes = []
    loop_times = []
    env_failure_episodes = []
    
    episode = 0
    while episode < test_num:
        cprint("\n" + "=" * 60, "magenta")
        cprint(f"📍 Episode {episode + 1}/{test_num}", "magenta", attrs=["bold"])
        cprint("=" * 60, "magenta")
        
        # 重置环境和模型
        real_env.reset()
        reset_func(model)
        
        # 自动生成指令（根据循环次数数组和模板，同时生成两种指令）
        # this_loop_time = loop_times_array[episode % len(loop_times_array)]
        cprint("\n⏳  请选择本轮的循环次数 from " + str(loop_times_array) + ": ", "yellow", attrs=["bold"], end="")
        this_loop_time = input().strip()
        cprint(f"🔄 当前循环次数: {this_loop_time}", "yellow")
        
        # 使用真机的指令生成方法（从模板生成，同时包含 instruction_int 和 instruction_sim）
        instruction_dict = generate_instruction_from_template(
            instruction_template, 
            this_loop_time
        )
        
        cprint(f"📝 instruction_int: {instruction_dict['instruction_int']}", "blue")
        cprint(f"📝 instruction_sim: {instruction_dict['instruction_sim']}", "blue")
        real_env.set_instruction(
            instruction=instruction_dict["instruction_sim"], # 这里没有full, 直接用sim
            instruction_int=instruction_dict["instruction_int"], 
            instruction_sim=instruction_dict["instruction_sim"]
        )
        
        # 等待用户准备场景
        input("\n⏸️  请摆放好场景，按 Enter 开始执行策略...")
        
        print("\n🚀 开始执行策略...\n")
        start_time = time.time()
        
        # 执行策略主循环
        success = False
        # trajectory = np.array([])  # 记录轨迹
        try:
            cprint("是否用数据集帧初始化？(y/n): ", "yellow", attrs=["bold"], end="")
            use_init_data = input().strip().lower() == 'y'
            if use_init_data:
                cprint("用多少帧数据初始化？(整数): ", "yellow", attrs=["bold"], end="")
                init_frame_num = int(input().strip())
                real_env._load_data_to_memory()
                updata_func = eval_function_decorator(policy_name, "updata_obs_real")
                for _ in range(init_frame_num):
                    observation = real_env.get_obs_dataset()
                    updata_func(model, observation)
                    real_env.take_action_dataset()
                    time.sleep(0.1)  # 过快can会爆掉

            while real_env.take_action_cnt < real_env.step_lim:
                # 获取观测
                observation = real_env.get_obs()
                
                # 执行策略（会自动调用 take_action）
                # 在policy/{policy_name}/deploy_policy.py中定义
                actions = eval_func(real_env, model, observation) # n * 7

                # 记录轨迹
                # trajectory = actions if trajectory.size == 0 else np.vstack((trajectory, actions))

        except KeyboardInterrupt:
            cprint("\n\n⚠️  用户中断执行当次测评", "red", attrs=["bold"])

            
        except Exception as e:
            cprint(f"\n\n❌ 执行出错: {e}", "red", attrs=["bold"])
            import traceback
            traceback.print_exc()
            
        elapsed_time = time.time() - start_time
        cprint(f"\n⏱️  执行时间: {elapsed_time:.2f}秒", "yellow")
        
        # 手动检查任务是否成功
        cprint("\n📊 检测到的循环次数是？", "yellow", attrs=["bold"])
        detected_loop_count = input().strip()
        try:
            detected_loop_count = int(detected_loop_count)
        except:
            detected_loop_count = 0
        loop_times.append(detected_loop_count)

        cprint("\n" + "-" * 60, "white")
        cprint("✔️  任务是否成功? (y/n): ", "yellow", attrs=["bold"], end="")
        success = input().strip().lower() == 'y'
        cprint("-" * 60, "white")

        is_env_fail = False
        if not success:
            cprint("是否是 没抓稳等 环境因素导致失败? (y/n): ", "yellow")
            is_env_fail = input().strip().lower() == 'y'

        # 记录结果 - 任务成功/失败
        if success:
            success_count += 1
            task_success_episodes.append(episode)
            cprint("\n✅ 任务成功!", "green", attrs=["bold"])
        else:
            task_failure_episodes.append(episode)
            cprint("\n❌ 任务失败", "red", attrs=["bold"])
        
        # 记录循环成功/失败（只有任务成功时才判断循环）
        if success:
            loop_succ = (this_loop_time == detected_loop_count)
            if loop_succ:
                loop_success_episodes.append(episode)
                cprint(f"✅ 循环成功: 期望 {this_loop_time} 次，检测到 {detected_loop_count} 次", "green")
            else:
                loop_failure_episodes.append(episode)
                cprint(f"❌ 循环失败: 期望 {this_loop_time} 次，检测到 {detected_loop_count} 次", "red")
        
        # 环境失败记录（不影响其他统计）
        if is_env_fail:
            cprint("⚠️  任务失败归因于环境因素", "yellow")
            env_failure_episodes.append(episode)
        
        real_env.test_num += 1
        if success:
            real_env.suc += 1
        
        # 显示当前统计
        task_success_rate = success_count / (episode + 1) * 100
        loop_success_rate = len(loop_success_episodes) / max(1, success_count) * 100 if success_count > 0 else 0
        cprint("\n" + "=" * 60, "cyan")
        cprint(f"📊 当前统计:", "cyan", attrs=["bold"])
        cprint(f"   任务成功: {success_count}/{episode + 1} ({task_success_rate:.1f}%)", "cyan")
        cprint(f"   循环成功: {len(loop_success_episodes)}/{success_count} ({loop_success_rate:.1f}%)", "cyan")
        cprint(f"   任务成功的轮次: {task_success_episodes}", "blue")
        cprint(f"   任务失败的轮次: {task_failure_episodes}", "blue")
        cprint(f"   循环成功的轮次: {loop_success_episodes}", "blue")
        cprint(f"   循环失败的轮次: {loop_failure_episodes}", "blue")
        cprint(f"   环境失败的轮次: {env_failure_episodes}", "blue")
        cprint("=" * 60, "cyan")
        
        episode += 1
        
        # 询问是否继续
        if episode < test_num:
            cprint("\n⏸️  继续下一轮测试? (y/n/q-退出): ", "yellow", attrs=["bold"], end="")
            cont = input().strip().lower()
            if cont == 'n':
                cprint("⏸️  暂停测试", "yellow")
                time.sleep(2)
            elif cont == 'q':
                cprint("🛑 退出测试", "red", attrs=["bold"])
                break
    
    # 最终结果
    task_success_rate = success_count / real_env.test_num * 100
    loop_success_rate = len(loop_success_episodes) / max(1, success_count) * 100 if success_count > 0 else 0
    cprint("\n" + "=" * 60, "green")
    cprint("📈 最终结果:", "green", attrs=["bold"])
    cprint(f"   任务成功率: {success_count}/{real_env.test_num} = {task_success_rate:.1f}%", "green")
    cprint(f"   循环成功率: {len(loop_success_episodes)}/{success_count} = {loop_success_rate:.1f}%", "green")
    cprint(f"   检测到的循环次数: {loop_times}", "cyan")
    cprint(f"   任务成功的轮次: {task_success_episodes}", "blue")
    cprint(f"   任务失败的轮次: {task_failure_episodes}", "blue")
    cprint(f"   循环成功的轮次: {loop_success_episodes}", "blue")
    cprint(f"   循环失败的轮次: {loop_failure_episodes}", "blue")
    cprint(f"   环境失败的轮次: {env_failure_episodes}", "blue")
    cprint("=" * 60 + "\n", "green")

    return success_count, task_success_episodes, task_failure_episodes, loop_success_episodes, loop_failure_episodes, env_failure_episodes, loop_times


def main(usr_args):
    """主函数"""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    exp_tag = usr_args["exp_tag"]
    test_num = usr_args.get('test_num', 609609)
    
    # 生成循环次数数组
    loop_times_min = usr_args.get('loop_times_min', 1)
    loop_times_max = usr_args.get('loop_times_max', 8)
    loop_times_array = list(range(loop_times_min, loop_times_max + 1))
    
    # 加载指令模板
    instruction_template = load_instruction_template(task_name)
    
    cprint("\n" + "=" * 60, "green")
    cprint("🤖 Piper 真机策略评估", "green", attrs=["bold"])
    cprint("=" * 60, "green")
    cprint(f"Policy: {policy_name}", "green")
    cprint(f"Task: {task_name}", "green")
    cprint(f"Task Config: {task_config}", "green")
    cprint(f"Checkpoint Setting: {ckpt_setting}", "green")
    cprint(f"Exp Tag: {exp_tag}", "green")
    cprint(f"Checkpoint Num: {checkpoint_num}", "green")
    cprint(f"Loop Times Array: {loop_times_array}", "cyan")
    cprint(f"Instruction Template: {'✓ Loaded' if instruction_template else '✗ Not Found (using default)'}", "yellow")
    cprint(f"Model: {usr_args.get('model_name', 'N/A')}", "cyan")
    cprint(f"Robot IP: {usr_args.get('robot_ip', 'can0')}", "cyan")
    cprint(f"测试轮数: {test_num}", "cyan")
    cprint("=" * 60 + "\n", "green")
    
    # 1. 加载模型
    cprint("📦 加载模型...", "yellow")
    get_model = eval_function_decorator(policy_name, "get_model_real")
    model = get_model(usr_args)
    cprint("✅ 模型加载完成\n", "green", attrs=["bold"])
    
    # 2. 初始化真机环境
    cprint("🏗️  初始化真机环境...", "yellow")

    data_dir = f"/home/dex/haoran/gello_software/data_processed/{task_name}/{task_config}/data/"
    try:
        assert os.path.exists(data_dir), f"用于初始化的数据目录不存在: {data_dir}"
        # 读取数据目录下的第一个h5文件
        h5_files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
        assert len(h5_files) > 0, f"数据目录下没有找到h5文件: {data_dir}"
        data_path = os.path.join(data_dir, h5_files[0])
        cprint(f"📂 使用数据文件 {data_path} 初始化环境", "blue")
        import h5py
        with h5py.File(data_path, 'r') as f:
            # 如果存在f['joint_state']['vector']，则使用它
            if 'joint_state' in f and 'vector' in f['joint_state']:
                init_joint_positions = f['joint_state']['vector'][0]
            else:
                init_joint_positions = f['joint_action']['vector'][0]

            usr_args['init_joint_positions'] = init_joint_positions
            print(f"初始位置: {init_joint_positions}")    
        # 读取h5文件，拿到joint_action/vector的第一个元素，作为初始位置
        # with h5py.File(data_path, 'r') as f:
        #     init_joint_positions = f['joint_action']['vector'][0]
        #     usr_args['init_joint_positions'] = init_joint_positions
        #     cprint(f"📍 初始位置: {init_joint_positions}", "blue")  
    except Exception as e:
        cprint(f"❌ 读取h5文件失败: {e}", "red", attrs=["bold"])
        cprint("是否继续? (y/n): ", "yellow", attrs=["bold"], end="")  
        cont = input().strip().lower()
        if cont != 'y':
            cprint("🛑 退出程序", "red", attrs=["bold"])
            raise e
        else:
            init_joint_positions = np.array([0.0, -0.0, -0.0, 0.7020077109336853] + 
                                            [-0.341080171311016, 0.33767049909606506, 
                                             0.006908714791052963, 0.09343911764866444, 
                                             0.7626252197300604, -0.8777070566538752, 
                                             0.04993957316833356, -0.2814281473516445, 
                                             -0.16155323341763803, 0.4014385403063156, 
                                             -0.3524181486645954, -1.1506913605633977, 
                                             -0.07861215198585017, 0.25988312787405904]) 
        
    
    real_env = RealEnv(
        policy=policy_name,
        robot_ip=usr_args.get('robot_ip', 'can0'),
        init_pos=init_joint_positions,
        step_lim=usr_args.get('step_lim', 800),
        img_size=tuple(usr_args.get('img_size', [640, 480]))
    )
    
    # 3. 执行评估
    success_count, task_success_episodes, task_failure_episodes, loop_success_episodes, loop_failure_episodes, env_failure_episodes, loop_times = eval_policy_realworld(
        policy_name=policy_name,
        model=model,
        real_env=real_env,
        args=usr_args,
        loop_times_array=loop_times_array,
        instruction_template=instruction_template,
        test_num=test_num
    )
    
    # 4. 保存结果
    save_dir = Path(usr_args.get('save_dir', './eval_result_realworld'))
    save_dir = save_dir / task_name / policy_name / f"{exp_tag}_{checkpoint_num}" / current_time
    save_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = save_dir / "result.txt"
    task_success_rate = success_count / max(real_env.test_num, 1) * 100
    loop_success_rate = len(loop_success_episodes) / max(1, success_count) * 100 if success_count > 0 else 0
    
    with open(result_file, 'w') as f:
        f.write(f"Timestamp: {current_time}\n\n")
        f.write(f"Policy: {policy_name}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Loop Times Array: {loop_times_array}\n\n")
        
        # Mission metrics
        f.write(f"=== Mission Metrics ===\n")
        f.write(f"Task Success Rate: {success_count}/{real_env.test_num} = {task_success_rate:.1f}%\n\n")
        f.write(f"Task Successful Episodes ({len(task_success_episodes)}):\n")
        f.write(", ".join(map(str, task_success_episodes)) + "\n\n")
        f.write(f"Task Failed Episodes ({len(task_failure_episodes)}):\n")
        f.write(", ".join(map(str, task_failure_episodes)) + "\n\n")
        
        # Loop metrics
        f.write(f"=== Loop Metrics ===\n")
        f.write(f"Loop Success Rate: {len(loop_success_episodes)}/{success_count} = {loop_success_rate:.1f}%\n\n")
        f.write(f"Loop Successful Episodes ({len(loop_success_episodes)}):\n")
        f.write(", ".join(map(str, loop_success_episodes)) + "\n\n")
        f.write(f"Loop Failed Episodes ({len(loop_failure_episodes)}):\n")
        f.write(", ".join(map(str, loop_failure_episodes)) + "\n\n")
        f.write(f"Detected Loop Times:\n")
        f.write(", ".join(map(str, loop_times)) + "\n\n")
        
        # Loop statistics
        if len(loop_times) > 0:
            mean_loop = np.mean(loop_times)
            std_loop = np.std(loop_times)
            f.write(f"Loop Times Mean: {mean_loop:.2f}, Std: {std_loop:.2f}\n\n")
        
        # Environment failure
        f.write(f"Environment Failed Episodes ({len(env_failure_episodes)}):\n")
        f.write(", ".join(map(str, env_failure_episodes)) + "\n")
    
    cprint(f"\n✅ 📁 结果已保存到: {result_file}", "green", attrs=["bold"])


def parse_args_and_config():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Piper 真机策略评估')
    parser.add_argument('--config', type=str, required=True, 
                        help='配置文件路径 (YAML)')
    parser.add_argument('--dont_stop', type=str, choices=['true', 'false'], default='false', 
                        help="Disable early stopping on success (true/false)")
    parser.add_argument('--overrides', nargs=argparse.REMAINDER,
                        help='覆盖配置参数')
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Add dont_stop to config (convert string to boolean)
    config["dont_stop"] = args.dont_stop.lower() == 'true'
    
    # 应用命令行覆盖
    if args.overrides:
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
        
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)
    
    return config


if __name__ == "__main__":
    cprint("🎬 let's go!!!", "cyan", attrs=["bold"])
    cprint("\n" + "=" * 60, "cyan")
    cprint("🤖 Piper Real World Policy Evaluation", "cyan", attrs=["bold"])
    cprint("=" * 60 + "\n", "cyan")
    
    try:
        usr_args = parse_args_and_config()
        main(usr_args)
    except KeyboardInterrupt:
        cprint("\n\n⚠️  程序被用户中断", "red", attrs=["bold"])
    except Exception as e:
        cprint(f"\n\n❌ 程序出错: {e}", "red", attrs=["bold"])
        import traceback
        traceback.print_exc()

import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser
import random

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def class_decorator(task_name):
    """
    根据任务名称导入对应的模块（.py文件）
    """
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        # TODO: 加入控制次数的参数去初始化环境
        env_instance = env_class()
    except Exception as e:
        print(e)
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

# 读取参数，初始化环境
def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)

def generate_loop_times_sequence(episode_num, loop_config):
    """
    预生成完整的 loop_times 序列文件
    
    Args:
        episode_num: 总的 episode 数量
        loop_config: loop 配置字典，包含 is_sequential, sequence_min, sequence_max 等信息
    
    Returns:
        loop_times_list: 预生成的 loop_times 列表
    """
    loop_times_list = []
    
    if loop_config.get("is_sequential", False):
        # 循环模式：按照 sequence_min 到 sequence_max 轮转
        sequence_min = loop_config.get("sequence_min", 1)
        sequence_max = loop_config.get("sequence_max", 8)
        sequence_range = sequence_max - sequence_min + 1
        
        for i in range(episode_num):
            loop_times = sequence_min + (i % sequence_range)
            loop_times_list.append(loop_times)
    
    elif loop_config.get("is_random", False):
        # 随机模式：每次随机选择
        for i in range(episode_num):
            loop_times = random.randint(loop_config.get("random_min", 1), 
                                        loop_config.get("random_max", 8))
            loop_times_list.append(loop_times)
    
    else:
        # 固定模式：所有 episode 都使用相同的 loop_times
        loop_times = loop_config.get("loop_times", 1)
        loop_times_list = [loop_times] * episode_num
    
    return loop_times_list

# 主要运行函数
def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list, loop_times_list = 0, 0, 0, [], []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True
        # 检查是否存在seed.txt文件，如果存在就从下一个未使用的seed开始
        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = seed_list[-1] + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")
        
        # 检查是否存在loop_times.txt文件
        loop_times_file_path = os.path.join(args["save_path"], "loop_times.txt")
        if os.path.exists(loop_times_file_path):
            # 如果文件存在，检查数量是否足够
            with open(loop_times_file_path, "r") as file:
                loop_times_content = file.read().split()
                if len(loop_times_content) != 0:
                    loop_times_list = [int(i) for i in loop_times_content]
            
            # 如果现有的loop_times数量不足，重新生成完整序列
            if len(loop_times_list) < args["episode_num"]:
                print(f"\033[93mExisting loop_times.txt has {len(loop_times_list)} entries, " 
                      f"but episode_num is {args['episode_num']}. Regenerating...\033[0m")
                loop_times_list = generate_loop_times_sequence(args["episode_num"], args["loop"])
                # 保存新的loop_times列表
                with open(loop_times_file_path, "w") as file:
                    for lt in loop_times_list:
                        file.write("%s " % lt)
        else:
            # 如果文件不存在，提前生成完整的loop_times序列
            print(f"\033[93mGenerating loop_times.txt with {args['episode_num']} entries...\033[0m")
            loop_times_list = generate_loop_times_sequence(args["episode_num"], args["loop"])
            # 保存loop_times列表
            with open(loop_times_file_path, "w") as file:
                for lt in loop_times_list:
                    file.write("%s " % lt)

        # ============= 收集数据主循环 =============
        while suc_num < args["episode_num"]:
            try:
                # 去到对应任务的py代码中，根据配置初始化环境
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                # 去到对应任务的py代码中，执行一次任务
                # 直接从预生成的loop_times_list中读取，确保均匀分布
                loop_times = loop_times_list[suc_num]
                TASK_ENV.play_once(loop_times=loop_times)

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            # 保存seed列表
            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)
            
            # 保存loop_times列表
            with open(os.path.join(args["save_path"], "loop_times.txt"), "w") as file:
                for lt in loop_times_list:
                    file.write("%s " % lt)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]
        
        # 读取保存的loop_times列表
        with open(os.path.join(args["save_path"], "loop_times.txt"), "r") as file:
            loop_times_content = file.read().split()
            loop_times_list = [int(i) for i in loop_times_content]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        # 新增严格pkl-seed匹配与skip逻辑
        used_seed = set()
        pkl2seed = {}
        unused_seed = set(seed_list)
        total_seed = list(seed_list)
        episode_num = args["episode_num"]

        for episode_idx in range(st_idx, episode_num):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")
            pkl_file = f"episode{episode_idx}.pkl"
            found = False
            for seed_try in total_seed:
                if seed_try in used_seed:
                    continue
                print(f"\033[36mProcessing episode {episode_idx}: try seed={seed_try}, pkl={pkl_file}\033[0m")
                try:
                    TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_try, **args)
                    traj_data = TASK_ENV.load_tran_data(episode_idx)
                    args["left_joint_path"] = traj_data["left_joint_path"]
                    args["right_joint_path"] = traj_data["right_joint_path"]
                    TASK_ENV.set_path_lst(args)
                except Exception as e:
                    print(f"\033[91mSeed/PKL mismatch or load error: episode {episode_idx}, seed={seed_try}, pkl={pkl_file}\033[0m")
                    print(f"\033[91mError details: {e}\033[0m")
                    print(f"\033[93m[SKIP] 跳过当前种子 seed={seed_try}，继续尝试下一个，但pkl不变\033[0m")
                    continue

                info_file_path = os.path.join(args["save_path"], "scene_info.json")
                if not os.path.exists(info_file_path):
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump({}, file, ensure_ascii=False)
                with open(info_file_path, "r", encoding="utf-8") as file:
                    info_db = json.load(file)

                try:
                    loop_times = loop_times_list[episode_idx]
                    info = TASK_ENV.play_once(loop_times)
                    info_db[f"episode_{episode_idx}"] = info
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump(info_db, file, ensure_ascii=False, indent=4)
                    TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                except Exception as e:
                    print(f"\033[91mPlay_once execution failed: episode {episode_idx}, seed={seed_try}, pkl={pkl_file}\033[0m")
                    print(f"\033[91mError details: {e}\033[0m")
                    TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                    print(f"\033[93m[SKIP] 跳过当前种子 seed={seed_try}，继续尝试下一个，但pkl不变\033[0m")
                    continue

                # Check success before merging to avoid creating bad files
                if TASK_ENV.check_success():
                    TASK_ENV.merge_pkl_to_hdf5_video()
                    TASK_ENV.save_loop_times(loop_times)
                    TASK_ENV.remove_data_cache()
                    used_seed.add(seed_try)
                    unused_seed.discard(seed_try)
                    pkl2seed[f"episode{episode_idx}"] = seed_try
                    found = True
                    break
                else:
                    TASK_ENV.remove_data_cache()
                    hdf5_file = os.path.join(args["save_path"], 'data', f'episode{episode_idx}.hdf5')
                    mp4_file = os.path.join(args["save_path"], 'video', f'episode{episode_idx}.mp4')
                    if os.path.exists(hdf5_file):
                        try:
                            os.remove(hdf5_file)
                            print(f"\033[93mDeleted failed HDF5: {hdf5_file}\033[0m")
                        except Exception as e:
                            print(f"\033[91mFailed to delete HDF5 {hdf5_file}: {e}\033[0m")
                    if os.path.exists(mp4_file):
                        try:
                            os.remove(mp4_file)
                            print(f"\033[93mDeleted failed MP4: {mp4_file}\033[0m")
                        except Exception as e:
                            print(f"\033[91mFailed to delete MP4 {mp4_file}: {e}\033[0m")
                    print(f"\033[91mCollect Error at episode {episode_idx} (seed={seed_try}), pkl=episode{episode_idx}.pkl, skipping seed.\033[0m")
                    print(f"\033[93m[SKIP] 跳过当前种子 seed={seed_try}，继续尝试下一个，但pkl不变\033[0m")
                    continue
            if not found:
                print(f"\033[91m[ERROR] episode{episode_idx} 没有找到可用种子，采集失败！\033[0m")
                pkl2seed[f"episode{episode_idx}"] = None

        # 统计并保存pkl-seed映射和未用seed
        stat_path = os.path.join(args["save_path"], "pkl_seed_stat.json")
        stat = {
            "pkl2seed": pkl2seed,
            "unused_seed": list(unused_seed)
        }
        with open(stat_path, "w", encoding="utf-8") as f:
            json.dump(stat, f, ensure_ascii=False, indent=2)

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    # 检测render情况
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)

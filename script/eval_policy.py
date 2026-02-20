import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    exp_tag = getattr(usr_args, "exp_tag", "test")


    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["dont_stop"] = usr_args.get("dont_stop", False)  # Pass dont_stop parameter

    # pi0不需要使用点云，强行关闭
    if usr_args.get("policy_name") == 'pi0':
        args['data_type']['pointcloud'] = False

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

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
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{exp_tag}_{str(checkpoint_num)}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
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
    print("\033[94mDont Stop Mode:\033[0m " + str(args.get("dont_stop", False)))
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = 100
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num, success_episodes, failure_episodes, loop_info, loop_succ_episode_list, loop_fail_episode_list, history_loop_time \
                    = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))
        
        # 添加成功和失败的episode序号记录
        file.write(f"mission metrics\n")
        file.write(f"\n\nSuccess Rate: {suc_num}/{test_num} = {suc_num/test_num*100:.1f}%\n")
        file.write(f"\nSuccessful Episodes ({len(success_episodes)}):\n")
        file.write(", ".join(map(str, success_episodes)))
        file.write(f"\n\nFailed Episodes ({len(failure_episodes)}):\n")
        file.write(", ".join(map(str, failure_episodes)))

        file.write(f"\n\nLoop Mission metrics\n")
        loop_succ_num = len(loop_succ_episode_list)
        file.write(f"\nLoop Success Rate: {loop_succ_num}/{test_num} = {loop_succ_num/test_num*100:.2f}%\n")
        file.write(f"\nLoop Successful Episodes ({len(loop_succ_episode_list)}):\n")
        file.write(", ".join(map(str, loop_succ_episode_list)))
        file.write(f"\n\nLoop Failed Episodes ({len(loop_fail_episode_list)}):\n")
        file.write(", ".join(map(str, loop_fail_episode_list)))
        file.write(f"\n\nHistory Loop Times:\n")
        file.write(", ".join(map(str, history_loop_time)))

        # 均值和方差
        if len(history_loop_time) > 0:
            mean_loop = np.mean(history_loop_time)
            std_loop = np.std(history_loop_time)
            file.write(f"\n\nHistory Loop Times Mean: {mean_loop:.2f}, Std: {std_loop:.2f}\n")

    print(f"Data has been saved to {file_path}")
    # return task_reward

    # ------------- loop --------------
    if len(loop_info) > 0:
        loop_file_path = os.path.join(save_dir, f"_loop_detail.txt")
        # 打开文件时指定utf-8编码，支持非ASCII字符
        with open(loop_file_path, "w", encoding="utf-8") as file:
            for idx, info in enumerate(loop_info):
                file.write(f"Episode {idx + 1}:\n")
                for key, value in info.items():
                    file.write(f"{key}: {value}\n")
                file.write("\n")
        print(f"Loop metric data has been saved to {loop_file_path}")
    # ------------- loop --------------

def eval_morse_sos_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")
    test_num=100
    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    task_succ_episode_list = []
    task_fail_episode_list = []
    loop_succ_episode_list = []
    loop_fail_episode_list = []
    history_loop_time = []

    # ------------- loop --------------
    print(f"use fixed loop times")
    loop_times = [9] # 为了统一，也用列表存储
    print(f"loop times: {loop_times}")

    # 一个装字典的列表
    loop_info: list[dict] = []
    # ------------- loop --------------

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            task_succ_episode_list.append(now_id)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])

        ############ choose loop times ############
        #     用循环列表来选择本次的loop times      #
        ###########################################
        this_loop_time = 9
        print(f"Episode {now_id} | CMD Loop times: {this_loop_time}")

        instruction = instruction.replace("[num]", str(this_loop_time))

        instruction_dict = {
                            "instruction_sim": instruction,
                            "instruction": instruction}

        TASK_ENV.set_instruction(instruction=instruction_dict)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            # input("Press Enter to start evaluation...")  # Wait for user input to start evaluation
            observation = TASK_ENV.get_obs() # 注意这里只拿了一帧的obs
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                if not args.get("dont_stop", False):
                    break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            task_fail_episode_list.append(now_id)
            print("\033[91mFail!\033[0m")

        now_id += 1

        # ------------- loop --------------
        _loop_info: dict = TASK_ENV.analyze_loop_metric()
        # 用作最后的保存
        if _loop_info is not None:
            loop_info.append(_loop_info)

        # 提取loop信息，以判断本次循环任务是否成功
        actual_loop_time = _loop_info.get("loop_times", None)
        loop_succ = this_loop_time == actual_loop_time

        print(f"\033[93mLoop Metric | Detected Loops: {_loop_info.get('loop_times', None)} | Expected Loops: {this_loop_time} | Result: {'Success' if loop_succ else 'Fail'}\033[0m")
        # ------------- loop --------------

        # ------------- wavelength --------------
        
        # 仅当loop任务成功时，才进行波长任务的评估
        wavelen_succ = False
        if loop_succ:
            # 提取wavelength信息
            """
                wave_info = {
                    "wavelengths": List[float],  # 所有检测到的波长
                    "wave_num": int,             # 检测到的波数量
                    "top_3_waves": {              # 波长最大的三个波
                        "order": List[int],      # 这三个波的编号
                        "length": List[float],   # 这三个波的波长
                    },
                    "lower_3_waves": {           # 波长最小的三个波
                        "order": List[int],      # 这三个波的编号
                        "length": List[float],   # 这三个波的波长
                    },
                    "middle_3_waves":          # 中间的三个波
                    {
                        "order": List[int],       # 该波的编号
                        "length": List[float],    # 该波的波长
                    }
            """
            wave_info = _loop_info.get("wave_info", None)
            top_3_waves = wave_info["top_3_waves"]
            lower_3_waves = wave_info["lower_3_waves"]
            middle_3_waves = wave_info["middle_3_waves"]
            morse_code = [] # 长度为9的list，存储每个波的长短信息，1为长波，0为短波
            
            for i in range(9):
                if i in top_3_waves["order"]:
                    morse_code.append(1)
                elif i in lower_3_waves["order"]:
                    morse_code.append(0)
                else:
                    # 中间的三个波，判断更接近哪个波长
                    middle_orders = middle_3_waves["order"]
                    middle_lengths = middle_3_waves["length"]
                    middle_wave_length = None
                    if i in middle_orders:
                        idx = middle_orders.index(i)
                        if idx < len(middle_lengths):
                            middle_wave_length = middle_lengths[idx]
                    if middle_wave_length is None:
                        # 无法匹配到中间波长度时，默认判为短波
                        morse_code.append(0)
                    else:
                        dis_to_top3 = np.mean([abs(middle_wave_length - l) for l in top_3_waves["length"]])
                        dis_to_lower3 = np.mean([abs(middle_wave_length - l) for l in lower_3_waves["length"]])
                        if dis_to_top3 < dis_to_lower3:
                            morse_code.append(1)
                        else:
                            morse_code.append(0)
                            
            # 对初始假定的三个最大波作NearestNeighbor修正
            # 若发现有波长的NN不是1，则改为0
            for i in range(3):
                wave_idx = top_3_waves["order"][i]
                # 以波长大小为标准，寻找该波的最近邻
                wave_length = top_3_waves["length"][i]
                all_lengths = wave_info["wavelengths"]
                # 计算距离并排除自身索引
                distances = [abs(wave_length - l) for l in all_lengths]
                if 0 <= wave_idx < len(distances):
                    distances[wave_idx] = float("inf")
                nn_idx = int(np.argmin(distances))
                # 保护性检查，避免越界或错误类型
                if 0 <= nn_idx < len(morse_code) and morse_code[nn_idx] != 1:
                    morse_code[wave_idx] = 0
            
            print(wave_info["wavelengths"])
            wavelen_succ = False
            
            # 判断波长任务是否成功
            wavelen_succ = (morse_code == [0,0,0,1,1,1,0,0,0])  # SOS的摩斯密码
            
            # 打印结果
            print(f"\033[93mWavelength Metric | Wave Num: {wave_info['wave_num']} | Detected Wavelengths: {wave_info['wavelengths']} | Detected Morse Code: {morse_code} | Result: {'Success' if wavelen_succ else 'Fail'}\033[0m")
            
        else:
            print(f"\033[93mWavelength Metric | Skipped due to Loop Mission Failure\033[0m")
        
        loop_succ = loop_succ and wavelen_succ
        
        history_loop_time.append(actual_loop_time)
        if loop_succ:
            loop_succ_episode_list.append(now_id - 1)
        else:
            loop_fail_episode_list.append(now_id - 1)
        
        # ------------- wavelength --------------

        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"mossion success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, task_succ_episode_list, task_fail_episode_list, loop_info, loop_succ_episode_list, loop_fail_episode_list, history_loop_time


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    
    if args['task_name'] == 'morse_sos':
        return eval_morse_sos_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=test_num,
                video_size=video_size,
                instruction_type=instruction_type)  
    
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    task_succ_episode_list = []
    task_fail_episode_list = []
    loop_succ_episode_list = []
    loop_fail_episode_list = []
    history_loop_time = []

    # ------------- loop --------------
    if "loop" in args:
        loop_times_min = args["loop"]["random_min"]
        loop_times_max = args["loop"]["random_max"]
        loop_times = []
        for i in range(loop_times_min, loop_times_max + 1):
            loop_times.append(i)
        print(f"loop times: {loop_times}")
    else:
        loop_times = None
        print(f"dont use loop")

    # 一个装字典的列表
    loop_info: list[dict] = []
    # ------------- loop --------------
    
    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:

        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            task_succ_episode_list.append(now_id)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]

        results = generate_episode_descriptions_inference(args["task_name"], episode_info_list, 1)
        instruction = results[0]["unseen"]
        
        instruction_int = results[0]["instructions_int"] if "instructions_int" in results[0] else None

        if loop_times is not None:
            this_loop_time = loop_times[now_id % len(loop_times)]
            print(f"Episode {now_id} | CMD Loop times: {this_loop_time}")
            instruction = instruction.replace("[num]", str(this_loop_time))
            instruction_int = instruction_int.replace("[num]", str(this_loop_time))
        else:
            instruction_int = None

        instruction_dict = {"instruction_int": instruction_int,
                            "instruction_sim": instruction,
                            "instruction": instruction}
        
        TASK_ENV.set_instruction(instruction=instruction_dict)  # set language instruction
        print(TASK_ENV.get_instruction())
        # raise Exception("1")

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            # input("Press Enter to start evaluation...")  # Wait for user input to start evaluation
            observation = TASK_ENV.get_obs() # 注意这里只拿了一帧的obs
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                if not args.get("dont_stop", False):
                    break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            task_fail_episode_list.append(now_id)
            print("\033[91mFail!\033[0m")

        now_id += 1

        # ------------- loop --------------
        _loop_info: dict = TASK_ENV.analyze_loop_metric()
        # 用作最后的保存
        if _loop_info is not None:
            loop_info.append(_loop_info)

        # 提取loop信息，以判断本次循环任务是否成功
        actual_loop_time = _loop_info.get("loop_times", None)

        if loop_times is not None:
            loop_succ = this_loop_time == actual_loop_time
            history_loop_time.append(actual_loop_time)
            if loop_succ:
                loop_succ_episode_list.append(now_id - 1)
            else:
                loop_fail_episode_list.append(now_id - 1)
        else:
            loop_succ = 0
            this_loop_time = 0
            history_loop_time.append(0)
            loop_fail_episode_list.append(now_id - 1)

        print(f"\033[93mLoop Metric | Detected Loops: {_loop_info.get('loop_times', None)} | Expected Loops: {this_loop_time} | Result: {'Success' if loop_succ else 'Fail'}\033[0m")
        # ------------- loop --------------

        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        # print(
        #     f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
        #     f"mossion success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        # )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, task_succ_episode_list, task_fail_episode_list, loop_info, loop_succ_episode_list, loop_fail_episode_list, history_loop_time


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dont_stop", type=str, choices=['true', 'false'], default='false', 
                        help="Disable early stopping on success (true/false)")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Add dont_stop to config (convert string to boolean)
    config["dont_stop"] = args.dont_stop.lower() == 'true'

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
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
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)

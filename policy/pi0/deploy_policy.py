import numpy as np
import torch
import dill
import os, sys
import time

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step)


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()


############## 以下是一套适合真机的 evaluate 环境代码 ################
##############   与上面的互相独立，可根据需要选择使用  ################
"""
部署真实机器人策略 (单头部相机 + 单臂7自由度)
与 deploy_policy.py 的主要区别：
- 只使用头部相机 (head_camera)
- 只使用右臂 (7 DoF)
"""
from pi_model_real import *

# Encode observation for the real robot model (single camera only)
def encode_obs_real(observation):
    """
    编码真实机器人的观察
    只使用头部相机
    
    Args:
        observation: 观察字典，包含:
            - observation["observation"]["head_camera"]["rgb"]: 头部相机图像
            - observation["joint_action"]["vector"]: 关节状态 (7 DoF)
    
    Returns:
        input_rgb: 头部相机图像
        input_state: 7维关节状态
    """
    # 只使用头部相机
    input_rgb = observation["observation"]["head_camera"]["rgb"]
    # 只使用右臂的7个自由度
    input_state = observation["joint_action"]["vector"]
    
    assert len(input_state) == 7, f"Expected 7 DoF for real robot, got {len(input_state)}"

    return input_rgb, input_state


def get_model_real(usr_args):
    """
    获取真实机器人的模型
    
    Args:
        usr_args: 用户参数字典，包含:
            - train_config_name: 训练配置名称 (如 "pi0_base_aloha_real_lora")
            - model_name: 模型名称
            - checkpoint_id: checkpoint ID
            - pi0_step: 推理步数
    
    Returns:
        PI0Real: 真实机器人的PI0模型实例
    """
    train_config_name, model_name, checkpoint_id, pi0_step = (
        usr_args["train_config_name"], 
        usr_args["model_name"],
        usr_args["checkpoint_id"], 
        usr_args["pi0_step"]
    )
    return PI0Real(train_config_name, model_name, checkpoint_id, pi0_step)


def eval_real(TASK_ENV, model: PI0Real, observation):
    """
    评估/推理函数
    
    Args:
        TASK_ENV: 任务环境
        model: PI0Real模型
        observation: 当前观察
    """
    # 第一次调用时设置语言指令
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    # 编码观察
    input_rgb, input_state = encode_obs_real(observation)
    # breakpoint()
    model.update_observation_window(input_rgb, input_state)

    # ======== Get Action ========
    actions = model.get_action()[:model.pi0_step]

    # 执行动作
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb, input_state = encode_obs_real(observation)
        model.update_observation_window(input_rgb, input_state)

        # pi0是推理一次执行50个动作，太tm快了，真机执行不过来，稍微等一下
        time.sleep(0.05)  

    # ============================


def reset_model_real(model):
    """重置模型的观察窗口"""
    model.reset_observation_windows()

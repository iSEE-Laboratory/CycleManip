# import packages and module here
import sys

import torch
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import main as hydra_main
import pathlib
from omegaconf import OmegaConf

import yaml
from datetime import datetime
import importlib
import time

from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

# 用于计算帧率的全局变量
_eval_real_last_time = None

sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))

from dp3_policy import *

def encode_obs(observation):  # Post-Process Observation
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    obs['point_cloud'] = observation['pointcloud']

    if "endpose" in observation:
        obs["endpose"] = observation["endpose"]

    if 'test_time_instruction' in observation:
        if isinstance(observation['test_time_instruction'], dict):
            for key in observation['test_time_instruction']:
                obs[key] = observation['test_time_instruction'][key]
        else:
            obs['instruction'] = observation['test_time_instruction']

    return obs

def get_model(usr_args):
    config_path = "./3D-Diffusion-Policy/diffusion_policy_3d/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    OmegaConf.set_struct(cfg, True)

    DP3_Model = DP3(cfg, usr_args)
    return DP3_Model


def eval(TASK_ENV, model: DP3, observation):
    # input("Press Enter to start evaluation...")  # Wait for user input to start evaluation
    obs = encode_obs(observation)  # Post-Process Observation
    # instruction = TASK_ENV.get_instruction()

    if len(model.env_runner.obs) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    actions = model.get_action()[4:]  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
         TASK_ENV.take_action(action)
         observation = TASK_ENV.get_obs()
         obs = encode_obs(observation)
         model.update_obs(obs)  # Update Observation, `update_obs` here can be modified
         # time.sleep(0.02)  # Control the action frequency
    """ START_TIME = 0
    END_TIME = 8
    for idx in range(START_TIME, END_TIME):
        action = actions[idx]
        TASK_ENV.take_action(action)

        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs) """

    return actions


def reset_model(model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.env_runner.reset_obs()



############## 以下是一套适合真机的 evaluate 环境代码 ################
##############   与上面的互相独立，可根据需要选择使用  ################


def encode_obs_real(observation):  # Post-Process Observation

    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    obs['point_cloud'] = observation['pointcloud']
    if 'instruction' in observation:
        obs['instruction'] = observation['instruction']
    if 'instruction_sim' in observation:
        obs['instruction_sim'] = observation['instruction_sim']
    if 'instruction_int' in observation:
        obs['instruction_int'] = observation['instruction_int']
    if 'endpose' in observation:
        obs['endpose'] = observation['endpose']
    if "short_state" in observation:
        obs["short_state"] = observation["short_state"]
    return obs


def updata_obs_real(model, observation):
    obs = encode_obs_real(observation)
    model.update_obs(obs)

def get_model_real(usr_args):
    config_path = "./3D-Diffusion-Policy/diffusion_policy_3d/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    OmegaConf.set_struct(cfg, True)

    DP3_Model = DP3(cfg, usr_args)
    return DP3_Model


def eval_real(TASK_ENV, model, observation):
    global _eval_real_last_time
    
    # print(observation)
    # input("Press Enter to start evaluation...")  # Wait for user input to start evaluation
    obs = encode_obs_real(observation)  # Post-Process Observation
    # instruction = TASK_ENV.get_instruction()

    if len(model.env_runner.obs) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    # t1 = time.time()
    actions = model.get_action()  # Get Action according to observation chunk
    # t2 = time.time()
    # print(f"diffusion time: {t2 - t1:.4f} seconds")
    # 去到 robot runner里的get_action

    # breakpoint()

    ah = len(actions)

    for i, action in enumerate(actions):  # Execute each step of the action
        # t2 = time.time()
        # print(action)
        TASK_ENV.take_action(action)

        # 计算帧率
        current_time = time.time()
        if _eval_real_last_time is not None:
            delta_time = current_time - _eval_real_last_time
            if delta_time > 0:
                fps = 1.0 / delta_time
                print(f"eval_real FPS: {fps:.2f}", end='\r')
        
        _eval_real_last_time = current_time

        ############## 夹爪 #############
        if i != ah - 1:
            time.sleep(0.08)  # Control the action frequency

        # if i != ah - 1:
        #     time.sleep(0.1)  # Control the action frequency
        
        ############# 夹爪 #############


        ############## 灵巧手 #############
        # 不用sleep
        ############## 灵巧手 #############


        ############## 人形 ##############
        # if i != ah - 1:
        #     time.sleep(0.08)  # Control the action frequency
        ############## 人形 ##############

        # print(f"take action time: {t3 - t2:.4f} seconds")
        observation = TASK_ENV.get_obs()
        # t4 = time.time()
        # print(f"get observation time: {t4 - t3:.4f} seconds")
        obs = encode_obs_real(observation)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified

        # t5 = time.time()
        # print(f"update observation time: {t5 - t4:.4f} seconds")

        # print(f"sleep time: {time.time() - t5:.4f} seconds")

        # print("-----")

    return actions


def reset_model_real(model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.env_runner.reset_obs()

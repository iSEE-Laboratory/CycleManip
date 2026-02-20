import os
from dex_vla.model_load_utils import load_model_for_eval

import torch
from torchvision import transforms
import cv2
from aloha_scripts.utils import *
import numpy as np
import time

from aloha_scripts.constants import FPS

from data_utils.dataset import set_seed
from einops import rearrange

import torch_utils as TorchUtils
# import matplotlib.pyplot as plt
import sys
from policy_heads import *
# from cv2 import aruco
from dex_vla.utils.image_processing_qwen2_vla import *
from paligemma_vla.utils.processing_paligemma_vla import *
from dex_vla.utils.processing_qwen2_vla import *
# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
from vla_policy import *
import copy

def preprocess_img(images: torch.Tensor):
    assert images.ndim == 4 and images.shape[1] == 3
    original_size = (320, 240)
    new_size = (448, 448)
    ratio = 0.95
    t1 = transforms.Resize(size=original_size, antialias=True)
    t2 = transforms.Resize(size=new_size, antialias=True)
    images = t1(images)
    images = images[...,
             int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
             int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
    images = t2(images)

    return images
class DexVLA:
    def __init__(self, policy_config, camera_names):
        super(DexVLA).__init__()
        self.camera_names = camera_names
        self.policy_config = policy_config
        self.task_name = policy_config["task_name"]
        self.state_path = policy_config["state_path"]
        model_base = policy_config["model_base"] # if policy_config["enable_lore"] else None
        model_path = policy_config["model_path"]
        print("Start Load the Model")
        policy = qwen2_vla_policy(policy_config)

        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=False,attn_implementation="default")
        self.vla_process = InternVL3Process(
            tokenizer=self.tokenizer,
            conv_template=self.policy.conv_template,
            camera_names=self.camera_names,
            num_image_token=self.policy.num_image_token
        )
        with open(self.state_path, 'rb') as f:
            self.stats = pickle.load(f)


    def pre_process(self, sample):
        stats = self.stats
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(sample[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        qpos_data = torch.from_numpy(sample["qpos"]).float()
        qpos_data = (qpos_data - stats["qpos_mean"]) / stats["qpos_std"]
        image_data = preprocess_img(image_data)
        qpos_data = qpos_data.unsqueeze(0)
        s = {
            'image': image_data,
            'state': qpos_data,
            'raw_lang': sample["raw_lang"],
        }
        return self.vla_process.preprocess(s)

    def get_action(self, obs=None):
        stats = self.stats
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
        # post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        batch = self.pre_process(obs)
        # actions = self.policy.sample_action(**batch).detach().cpu().numpy()
        actions = self.policy.sample_action(**batch).detach().cpu().to(torch.float32).numpy()
        actions = np.squeeze(actions, axis=0)
        actions = post_process(actions)
        return actions


task_prompt = {
    "place_object_scale": "Use one arm to grab the object and put it on the scale.",
    "place_phone_stand": "Your task is to assist the robot in placing a phone onto a phone stand, both of which are randomly positioned on the desk at initialization. You will be provided with images of the desk from different angles to help determine the positions of the phone and phone stand, and to plan the necessary actions to accomplish the placement.",
    "blocks_stack_three": "Your task is to assist the robot in stacking three cubes on the desk in a specific order: red at the bottom, green in the middle, and blue on top. The cubes will be randomly placed on the desk at initialization. You will be provided with images from different angles to help determine the positions of the cubes and to plan the necessary actions to accomplish the stacking task.",
    "blocks_ranking_rgb": "Your task is to assist the robot in sorting three cubes on the desk so that they are arranged in the order of red, green, and blue from left to right. The cubes will be randomly placed on the desk at initialization. You will be provided with images from different angles to help determine the positions of the cubes and to plan the necessary actions to accomplish the sorting task.",
    "dual_shoes_place": "Your task is to assist the robot in placing two shoes into a shoe box, with the shoes oriented to the left. The shoes will be randomly placed on the floor or a surface at initialization, while the shoe box is fixed at a certain location. You will be provided with images from different angles to help determine the positions of the shoes and the shoe box, and to plan the necessary actions to accomplish the task.",
    "put_bottles_dustbin": "Your task is to assist the robot in putting three bottles into the trash bin. The bottles are randomly placed on the desk at initialization. You will be provided with images of the desk from different angles to help determine the positions of the bottles and the trash bin, and to plan the necessary actions to accomplish the task.",
}
task_reasoning = {
    "place_object_scale": 0,
    "place_phone_stand": 1
}
all_reasoning = [
    ["Pick up the object.","Place the object onto the scale."],
    [],
]

def encode_obs(observation):  # Post-Process Observation
    """
    Process input data for VLA model。
    """
    obs = observation
    cam_high = obs["observation"]["head_camera"]["rgb"]
    cam_left = obs["observation"]["left_camera"]["rgb"]
    cam_right = obs["observation"]["right_camera"]["rgb"]
    qpos = (observation["joint_action"]["left_arm"] + [observation["joint_action"]["left_gripper"]] +
            observation["joint_action"]["right_arm"] + [observation["joint_action"]["right_gripper"]])
    #print("Check:", qpos)
    qpos = np.array(qpos)
    #print("Check:", qpos)
    return {
        "cam_high": cam_high,
        "cam_left": cam_left,
        "cam_right": cam_right,
        "qpos": qpos,
    }


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    """
    加载模型
    """
    camera_names = ['cam_high', 'cam_left', 'cam_right']
    task_name = usr_args["task_name"]
    model_path = usr_args["model_path"]
    action_head = 'dit_diffusion_policy'  # 'unet_diffusion_policy'
    model_size = '2B'
    policy_config = {
        "model_path": model_path,
        "pretrain_path": dit_path,
        "enable_lora": True,
        "conv_mode": "pythia",
        "temp_agg": False,
        "action_head": action_head,
        'model_size': model_size,
        'save_model': False,
        'control_mode': 'absolute',  # absolute
        "DexVLA": False,
        "history_image_length": 1,
        "ema": False,
        "camera_views": 3,
    }
    model = DexVLA(policy_config, camera_names)
    return model  # return your policy model


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = task_prompt[model.task_name]
    obs.update({"raw_lang": str(instruction)})
    len_traj = 1000
    reasonings = sub_reasons = [all_reasoning[task_reasoning[task_name]][0]] * int(len_traj/2) + [all_reasoning[task_reasoning[task_name]][1]] * (len_traj - int(len_traj/2))
    obs.update({"reasonings": str(reasonings)})
    # print("******************************")
    actions = model.get_action(obs)  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        # TASK_ENV.take_one_step_action(action)
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass

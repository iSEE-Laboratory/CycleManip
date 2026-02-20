import os
from dex_vla.model_load_utils import load_model_for_eval
import torch
from torchvision import transforms
import cv2
from aloha_scripts.utils import *
import numpy as np
import time
from aloha_scripts.constants import FPS
from data_utils.dataset import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, \
    postprocess_base_action  # helper functions
from einops import rearrange
import torch_utils as TorchUtils
# import matplotlib.pyplot as plt
import sys
from policy_heads import *
from paligemma_vla.models.modeling_paligemma_vla import *
from vla_policy import *
import copy
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from smart_eval_agilex_v2 import eval_bc


class FakeRobotEnv():
    """Fake robot environment used for testing model evaluation, please replace this to your real environment."""
    def __init__(self, episode_name=None):
        self.real_data = False
        self.time_step = 0
        if episode_name is not None:
            import h5py
            data = h5py.File(episode_name, 'r')
            self.states = data['observations']['qpos']
            self.images = data['observations']['images']
            self.real_data = True
            pass

    def step(self, action, mode=''):
        print("Execute action successfully!!!")

    def reset(self):
        print("Reset to home position.")

    def get_obs(self):
        if self.real_data:
            obs = {}
            for k,v in self.images.items():
                if 'front' in k:
                    k = k.replace('front', 'bottom')
                if 'high' in k:
                    k = k.replace('high', 'top')
                obs[k] = v[self.time_step]
            states = self.states[self.time_step]
            self.time_step += 1
        else:
            img = cv2.imread('./test.png')
            obs = {
                'cam_left_wrist': img,
                'cam_right_wrist': img,
                'cam_bottom': img,
                'cam_top': img,
            }
            states = np.zeros(14)
        return {
            'images': obs,
            'qpos': states,
        }

if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    root = "/media/rl/MAD-1"

    action_head = 'dit_diffusion_policy'  # 'unet_diffusion_policy'
    model_size = '2B'
    policy_config = {

        "model_path": "/media/rl/HDD/data/multi_head_train_results/aloha_qwen2_vla/qwen2_vl_2B/qwen2_vl_3_cameras_standard_folding_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_3w/checkpoint-30000",

        # "model_path": f"/media/rl/HDD/data/multi_head_train_results/aloha_qwen2_vla/paligemma_3B/paligemma_aloha_all_1_17_combine_constant_pretrain_Non_EMA_DIT_H_full_param/checkpoint-100000",

        # "model_base": f"/home/eai
        # /Downloads/Qwen2-VL-{model_size}-Instruct",
        # "model_base": "/home/eai/Documents/wjj/evaluate/vla-paligemma-3b-pt-224",
        "model_base": None,
        # "pretrain_dit_path": f"/home/eai/Documents/ljm/scaledp/filmresnet50_with_lang_sub_reason/fold_t_shirt_easy_version_1212_DiT-L_320_240_32_1e-4_numsteps_100000_scaledp_429traj_12_16/policy_step_100000.ckpt",
        "pretrain_dit_path": None,
        # "pretrain_path": '/media/eai/PSSD-6/wjj/results/aloha/Qwen2_vla-v0-robot-action-38k_droid_pretrain_lora_all_wo_film/checkpoint-40000',
        # "pretrain_path": "/home/eai/Documents/wjj/results/qwen2_vl_all_data_1200_align_frozen_dit_lora_substep/checkpoint-40000",
        # "pretrain_path": f"{root}/wjj/qwen2_vla_aloha/qwen2_vl_all_data_1200_align_frozen_dit_lora_substep_chunk_50/checkpoint-40000",
        "pretrain_path": None,
        "enable_lora": True,
        "conv_mode": "pythia",
        "temp_agg": False,
        "action_head": action_head,
        'model_size': model_size,
        'save_model': False,
        'control_mode': 'absolute',  # absolute
        "tinyvla": False,
        "history_image_length": 1,
        "ema": False,
        "camera_views": 3,
    }
    global im_size
    global save_dir
    save_dir = 'traj_2'
    im_size = 320  # default 480
    select_one = False  # select one embedding or using all
    raw_lang = 'I am hungry, is there anything I can eat?'
    raw_lang = 'I want to paste a poster, can you help me?'
    raw_lang = 'I want a container to put water in, can you help me?'
    # raw_lang = 'Upright the tipped-over pot.'
    # raw_lang = 'Put the cup on the tea table and pour tea into the cup'
    # raw_lang = 'Put the white car into the drawer.'
    # raw_lang = "Solve the equation on the table."
    raw_lang = "Arrange the objects according to their types."
    raw_lang = 'Classifying all objects and place to corresponding positions.'
    # raw_lang = 'Upright the tipped-over pot.'
    # raw_lang = "put the purple cube into the blue box."
    # raw_lang = "put the purple cube into the yellow box."
    # raw_lang = 'Upright the tipped-over yellow box.'
    # raw_lang = 'Put the cup onto the plate.'
    raw_lang = 'Place the toy spiderman into top drawer.'
    # raw_lang = "I want to make tea. Where is the pot?"
    # raw_lang = 'Clean the table.'
    # raw_lang = 'Store the tennis ball into the bag.'
    raw_lang = 'Sorting the tablewares and rubbish on the table.'
    # raw_lang = 'What is the object on the table?'
    # raw_lang = 'Arrange paper cups on the table.'
    # raw_lang = "Solve the rubik's cub."
    # raw_lang = 'Can you help me pack these stuffs?'
    raw_lang = 'Fold t-shirt on the table.'
    # raw_lang = "Serve a cup of coffee."
    # raw_lang = "Organize the bottles on the table."
    raw_lang = 'The crumpled shirts are in the basket. Pick it and fold it.'

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # sys.path.insert(0, "/home/eai/Dev-Code/mirocs")
    # from run.agilex_robot_env import AgilexRobot
    # agilex_bot = AgilexRobot()

    agilex_bot = FakeRobotEnv("/media/rl/HDD/data/data/aloha_data/4_cameras_aloha/fold_shirt_wjj1213_meeting_room/episode_0.hdf5")

    print('Already connected!!!!!!')
    # while True:
    #     obs = agilex_bot.get_obs()

    if 'paligemma' in policy_config['model_path'].lower():
        print(f">>>>>>>>>>>>>paligemma<<<<<<<<<<<<<<<")
        if 'lora' in policy_config['model_path'].lower():
            policy_config["model_base"] = "/home/eai/Documents/wjj/evaluate/vla-paligemma-3b-pt-224"

        policy = paligemma_vla_policy(policy_config)
    else:
        print(f">>>>>>>>>>>>>qwen2vl<<<<<<<<<<<<<<<")
        if 'lora' in policy_config['model_path'].lower():
            policy_config["model_base"] = f"/home/eai/Documents/wjj/Qwen2-VL-{model_size}-Instruct"

        policy = qwen2_vla_policy(policy_config)

    print(policy.policy)

    eval_bc(policy, agilex_bot, policy_config, raw_lang=raw_lang)

    print()
    exit()


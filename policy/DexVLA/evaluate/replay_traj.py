import os

import torch
from torchvision import transforms
import cv2

import numpy as np
import time
from time import sleep
import torch_utils as TorchUtils
import h5py
import sys

# from cv2 import aruco

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# import copy
# from data_utils.dataset import preprocess, preprocess_multimodal

def convert_actions(pred_action):
    # pred_action = torch.from_numpy(actions)
    # pred_action = actions.squeeze(0)
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)

    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
    # print(f'cur_xyz size: {cur_xyz.shape}')
    # print(f'cur_euler size: {cur_euler.shape}')
    # print(f'cur_gripper size: {cur_gripper.shape}')
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
    # print(f'4. pred_action size: {pred_action.shape}')
    print(f'4. after convert pred_action: {pred_action}')

    return pred_action

def eval_bc(deploy_env, policy_config, num_rollouts=1, raw_lang=None):

    with h5py.File(policy_config['data_path'], 'r') as f:
        actions = f['action'][()]
        # language = f['language_raw'][0].decode('utf-8')
        # language = ''
    for a in actions:
        obs = deploy_env.get_observation()
        cur_cartesian_position = np.array(obs['robot_state']['cartesian_position'])
        cur_gripper_position = np.expand_dims(np.array(obs['robot_state']['gripper_position']), axis=0)
        cur_state_np_raw = np.concatenate((cur_cartesian_position, cur_gripper_position))
        print(cur_state_np_raw)
        # print(f"Task is {language}")
        a = convert_actions(a)
        # a[5:] = cur_state_np_raw[5:]
        action_info = deploy_env.step(a)
        sleep(0.5)

    return


if __name__ == '__main__':
    policy_config = {
        'data_path': "/mnt/HDD/droid/h5_format_data/4types_pig_cyan_trunk_hex_key_gloves_480_640/4types_pig_cyan_trunk_hex_key_gloves_480_640_succ_t0001_s-0-0/episode_20.hdf5",
    }


    sys.path.insert(0, "/home/eai/Dev-Code/droid")
    from droid.robot_env import RobotEnv

    # from pynput import keyboard

    policy_timestep_filtering_kwargs = {'action_space': 'cartesian_position', 'gripper_action_space': 'position',
                                        'robot_state_keys': ['cartesian_position', 'gripper_position',
                                                             'joint_positions']}
    # resolution (w, h)
    # todo H W or W H?

    policy_camera_kwargs = {
        'hand_camera': {'image': True, 'concatenate_images': False, 'resolution': (480, 270), 'resize_func': 'cv2'},
        'varied_camera': {'image': True, 'concatenate_images': False, 'resolution': (480, 270), 'resize_func': 'cv2'}}

    deploy_env = RobotEnv(
        action_space=policy_timestep_filtering_kwargs["action_space"],
        gripper_action_space=policy_timestep_filtering_kwargs["gripper_action_space"],
        camera_kwargs=policy_camera_kwargs
    )

    deploy_env._robot.establish_connection()
    deploy_env.camera_reader.set_trajectory_mode()

    eval_bc(deploy_env, policy_config)



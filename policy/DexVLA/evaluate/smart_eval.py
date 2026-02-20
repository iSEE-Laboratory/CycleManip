import os
from dex_vla.model_load_utils import load_model_for_eval

import torch
from torchvision import transforms
import cv2

import numpy as np
import time

from aloha_scripts.constants import FPS

from data_utils.utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, \
    postprocess_base_action  # helper functions
from PIL import Image
from qwen_vl_utils import fetch_image
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM
from einops import rearrange
import torch_utils as TorchUtils
# import matplotlib.pyplot as plt
import sys
from policy_heads import *
# from cv2 import aruco
from dex_vla.utils.image_processing_qwen2_vla import *
from dex_vla.utils.processing_qwen2_vla import *
# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

import copy


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    return curr_image


def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp


def get_obs(deplot_env_obs, stats):
    # obs['front'], ['wrist_1'], ['state']
    cur_traj_data = dict()
    # (480, 270, 4)
    cur_right_rgb = deplot_env_obs['image']['21729895_left']  # camera_extrinsics image
    cur_left_rgb = deplot_env_obs['image']['29392465_left']  # camera_extrinsics image
    cur_wrist_rgb = deplot_env_obs['image']['18361939_left']  # camera_extrinsics image
    cur_wrist_rgb = cv2.resize(cur_wrist_rgb, (480, 270))

    w, h = 480, 270
    center = (w // 2, h // 2)
    angle = 180
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    cur_wrist_rgb = cv2.warpAffine(cur_wrist_rgb, M, (w, h))

    # [..., ::-1]
    # cur_front_rgb = cv2.cvtColor(cur_front_rgb, cv2.COLOR_BGRA2BGR)[..., ::-1]
    # cur_wrist_rgb = cv2.cvtColor(cur_wrist_rgb, cv2.COLOR_BGRA2BGR)[..., ::-1]

    cur_right_rgb = cv2.cvtColor(cur_right_rgb, cv2.COLOR_BGRA2BGR)
    cur_left_rgb = cv2.cvtColor(cur_left_rgb, cv2.COLOR_BGRA2BGR)
    cur_wrist_rgb = cv2.cvtColor(cur_wrist_rgb, cv2.COLOR_BGRA2BGR)

    # cur_front_rgb = cv2.cvtColor(cur_front_rgb, cv2.COLOR_BGRA2RGB)
    # cur_wrist_rgb = cv2.cvtColor(cur_wrist_rgb, cv2.COLOR_BGRA2RGB)
    # cv2.imshow('cur_rgb', cv2.hconcat([cur_left_rgb, cur_right_rgb, cur_wrist_rgb]))
    # cv2.waitKey(1)

    cur_right_depth = np.zeros_like(cur_right_rgb) - 1.0
    cur_right_depth = cur_right_depth[..., :1]
    cur_left_depth = np.zeros_like(cur_left_rgb) - 1.0
    cur_left_depth = cur_left_depth[..., :1]

    cur_cartesian_position = np.array(deplot_env_obs['robot_state']['cartesian_position'])
    # cur_cartesian_position = pre_process(cur_cartesian_position, 'tcp_pose', stats)

    cur_gripper_position = np.expand_dims(np.array(deplot_env_obs['robot_state']['gripper_position']), axis=0)
    # cur_gripper_position = pre_process(cur_gripper_position, 'gripper_pose', stats)

    cur_state_np_raw = np.concatenate((cur_cartesian_position, cur_gripper_position))

    cur_state_np = pre_process(cur_state_np_raw, 'qpos', stats)

    # [128, 128, 3] np array
    right_rgb_img = cur_right_rgb  # deplot_env_obs['front']
    right_depth_img = cur_right_depth
    left_rgb_img = cur_left_rgb  # deplot_env_obs['wrist_1']
    left_depth_img = cur_left_depth
    wrist_rgb_img = cur_wrist_rgb

    cur_state = cur_state_np  # deplot_env_obs['state']
    cur_state = np.expand_dims(cur_state, axis=0)

    # [2, 1, 128, 128, 3]
    # [2, 480, 480, 3]
    traj_rgb_np = np.array([left_rgb_img, right_rgb_img, wrist_rgb_img])

    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))
    # print(f'1. traj_rgb_np size: {traj_rgb_np.shape}')
    # l, n, c, h, w = traj_rgb_np.shape
    # traj_rgb_np = np.reshape(traj_rgb_np, (l, n*c, h, w))

    traj_depth_np = np.array([right_depth_img, left_depth_img])
    traj_depth_np = np.expand_dims(traj_depth_np, axis=1)
    traj_depth_np = np.transpose(traj_depth_np, (1, 0, 4, 2, 3))
    # print(f'1. traj_depth_np size: {traj_depth_np.shape}')
    # l, n, c, h, w = traj_depth_np.shape
    # traj_depth_np = np.reshape(traj_depth_np, (l, n*c, h, w))

    print("#" * 50)
    print(traj_rgb_np.shape)
    traj_rgb_np = np.array([[cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB) for img in traj_rgb_np[0]]])

    if im_size == 320:  # resize to 320
        traj_rgb_np = np.array([[cv2.resize(img, (320, 240)) for img in traj_rgb_np[0]]])

    traj_rgb_np = np.transpose(traj_rgb_np, (0, 1, 4, 2, 3))
    return cur_state_np_raw, cur_state, traj_rgb_np, traj_depth_np


def time_ms():
    return time.time_ns() // 1_000_000


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


class qwen2_vla_policy:
    def __init__(self, policy_config, data_args=None):
        super(qwen2_vla_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        # self.conv = conv_templates[policy_config['conv_mode']].copy()
        model_base = policy_config["model_base"] if policy_config[
            'enable_lora'] else None
        model_path = policy_config["model_path"]

        self.tokenizer, self.policy, self.multimodal_processor, self.context_len = load_model_for_eval(model_path=model_path,
                                                                                                    model_base=model_base, policy_config=policy_config)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["[SOA]"]})

        self.config = AutoConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)
    def datastruct_droid2qwen2vla(self, raw_lang):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": None,
                    },
                    {
                        "type": "image",
                        "image": None,
                    },
                    {
                        "type": "image",
                        "image": None,
                    },
                    {"type": "text", "text": f""},
                ],
            },
            # {"role": "assistant", "content": f''},
        ]

        messages[0]['content'][-1]['text'] = raw_lang
        # messages[1]['content'] = sample['reasoning'] + "Next action:"
        # print(sample['obs']['raw_language'].decode('utf-8'))
        return messages
    def process_batch_to_qwen2_vla(self, curr_image, robo_state, raw_lang):

        if len(curr_image.shape) == 5:  # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        messages = self.datastruct_droid2qwen2vla(raw_lang)
        image_data = torch.chunk(curr_image, curr_image.shape[0], dim=0)  # left, right ,wrist
        image_list = []
        for i, each in enumerate(image_data):
            ele = {
                # "resized_height": None,
                # "resized_width": None
            }
            each = Image.fromarray(each.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
            ele['image'] = each
            if i == 2:
                ele['resized_height'] = 56
                ele['resized_width'] = 56
            else:
                ele['resized_height'] = 240
                ele['resized_width'] = 320
            each = fetch_image(ele)
            image_list.append(torch.from_numpy(np.array(each)))
        # TODO RESIZE
        # image_data = image_data / 255.0
        image_data = image_list
        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # image_inputs, video_inputs = process_vision_info(dataset)
        # text = text[:-23]
        video_inputs = None
        model_inputs = self.multimodal_processor(
            text=text,
            images=image_data,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        data_dict = dict(states=robo_state)
        for k, v in model_inputs.items():
            data_dict[k] = v
        return data_dict


def eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=None, select_one=False):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)

    rand_crop_resize = True
    model_config = policy.config.policy_head_config

    temporal_agg = policy_config['temp_agg']
    action_dim = getattr(model_config, 'input_dim', 10)
    state_dim = getattr(model_config, 'state_dim', 7)

    policy.policy.eval()

    import pickle
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_config["action_head"].lower() == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif 'diffusion' in policy_config["action_head"] or 'vqbet' in policy_config["action_head"]:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    env = deploy_env

    query_frequency = 16
    if temporal_agg:
        query_frequency = 1
        num_queries = int(query_frequency)
    else:
        query_frequency = int(query_frequency / 2)
        num_queries = query_frequency
        from collections import deque
        action_queue = deque(maxlen=num_queries)


    max_timesteps = int(1000 * 10)  # may increase for real-world tasks

    for rollout_id in range(1000):

        rollout_id += 0

        env.reset(randomize=False)

        print(f"env has reset!")

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim],
                                           dtype=torch.bfloat16).cuda()
            # print(f'all_time_actions size: {all_time_actions.size()}')

        # robot_state_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        robot_state_history = np.zeros((max_timesteps, state_dim))
        image_list = []  # for visualization
        depth_list = []

        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            for t in range(max_timesteps):
                if t % 100 == 1:
                    a = input("q means next eval:")
                    if a== 'q':
                        env.reset(randomize=False)
                        lang_in = input("Input the raw_lang(q and enter mean using default):")
                        if lang_in != 'q' or lang_in != '':
                            raw_lang = lang_in
                            print(raw_lang)

                        break

                time1 = time.time()

                obs = deploy_env.get_observation()

                cur_state_np_raw, robot_state, traj_rgb_np, traj_depth_np = get_obs(obs, stats)
                print("curent robot state!!!!!!!!!!!!!!1",obs['robot_state']['cartesian_position'])

                image_list.append(traj_rgb_np)
                depth_list.append(traj_depth_np)
                robot_state_history[t] = cur_state_np_raw

                robot_state = torch.from_numpy(robot_state).float().cuda()

                # todo add resize&crop to wrist camera
                if t % query_frequency == 0:
                    curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
                    if rand_crop_resize:
                        print('rand crop resize is used!')
                        original_size = curr_image.shape[-2:]
                        ratio = 0.95
                        curr_image = curr_image[...,
                                     int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)
                        curr_image = curr_image.unsqueeze(0)

                # control_timestamps["policy_start"] = time_ms()
                if t == 0:
                    # warm up
                    for _ in range(2):
                        batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
                        if policy_config['tinyvla']:
                            policy.policy.evaluate_tinyvla(**batch, is_eval=True, select_one=select_one, tokenizer=policy.tokenizer)
                        else:
                            all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, select_one=select_one, tokenizer=policy.tokenizer)
                            print("*" * 50)
                            print(outputs)

                    print('network warm up done')
                    time1 = time.time()

                if t % query_frequency == 0:
                    batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
                    if policy_config['tinyvla']:
                        all_actions, outputs = policy.policy.evaluate_tinyvla(**batch, is_eval=True, select_one=select_one, tokenizer=policy.tokenizer)
                    else:
                        all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, select_one=select_one, tokenizer=policy.tokenizer)
                    if not temporal_agg:
                        action_queue.extend(
                            torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:num_queries])

                if temporal_agg:
                    print(f"all_actions: {all_actions.size()}")
                    print(f"all_time_actions: {all_time_actions.size()}")
                    print(f"t: {t}, num_queries:{num_queries}")
                    all_time_actions[[t], t:t + num_queries] = all_actions[:, :num_queries, :]
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = action_queue.popleft()


                print(f"raw action size: {raw_action.size()}")
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                action = post_process(raw_action)
                print(f"after post_process action size: {action.shape}")
                # target_qpos = action

                action = convert_actions(action.squeeze())
                print(f'step {t}, pred action: {outputs}{action}')
                action_info = deploy_env.step(action)

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            # plt.close()

    return


if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    sys.path.insert(0, "/home/eai/Dev-Code/droid")
    from droid.robot_env import RobotEnv
    policy_timestep_filtering_kwargs = {'action_space': 'cartesian_position', 'gripper_action_space': 'position',
                                        'robot_state_keys': ['cartesian_position', 'gripper_position',
                                                             'joint_positions']}
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

    action_head = 'dit_diffusion_policy'  # unet_diffusion_policy
    model_size = '2B'
    policy_config = {
        # "model_path": f"/media/eai/WJJ1T/droid/results/dex_vla/{model_size}/llavaPythia-v0-robot-action-10_7_math_reasoning_lora_all_film_residual/checkpoint-40000",
        # "model_path": f"/media/eai/PSSD-6/wjj/results/multi_head/Qwen2_vla-v0-robot-action-10_13_reasoning_8mt_lora_all_film_residual_pretrain/checkpoint-30000",
        # "model_path":f"/media/eai/SanDisk/wjj/7B/Qwen2_vla-v0-robot-action-10_31_reasoning_bin_picking_lora_all_film_residual/checkpoint-40000",
        # "model_path": "/media/eai/ExtremePro/wjj/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_film_residual_reasoning_38kplus1k_pretrain_4_epoch/checkpoint-45000",
        # "model_path":f"/media/eai/PSSD-6/wjj/results/multi_head/Qwen2_vla-v0-robot-action-10_31_reasoning_bin_picking_lora_all_film_residual_pretrain_1_epoch/checkpoint-40000",
        # "model_path": "/home/eai/wjj/72B_weights/72B/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_lr/checkpoint-40000",
        # "model_path":f"/media/eai/PSSD-6/wjj/results/multi_head/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_film_residual_reasoning_38kplus1k_pretrain_1_epoch_reinit/checkpoint-45000",
        # "model_path": '/media/eai/SanDisk/wjj/7B/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_film_residual_reasoning_38kplus1k_pretrain_1_epoch/checkpoint-45000', # 7B
        # "model_path": "/media/eai/PSSD-6/wjj/results/multi_head/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_film_residual_reasoning_38kplus1k_pretrain_1_epoch_reinit/checkpoint-45000",

        # 2B unet
        # "model_path": f"/media/eai/MAD-1/wjj/2B/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_film_residual/checkpoint-45000",  # w reasoning, wo pretrain, Qwen2-vl 2B
        # "model_path": "/media/eai/MAD-1/wjj/unet_head_qwen2_vla/2B/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_wo_reasoning_tinyvla/checkpoint-45000",  # TinyVLA QWen2-VLA 2B
        # "model_path": "/media/eai/PSSD-6/wjj/results/multi_head/2B/Qwen2_vla-v0-robot-action-11_1_all_lora_gt_reasoning_embedding/checkpoint-45000", # train w groundtruth reasoning embedding
        # "model_path": "/media/eai/PSSD-6/wjj/results/multi_head/2B/Qwen2_vla-v0-robot-action-11_1_all_lora_gt_reasoning_embedding_using_all/checkpoint-45000",# train wgt reasoning embedding and hidden embedding

        # 2B dit
        # "model_path": "/media/eai/MAD-1/wjj/dit_head_qwen2_vla/2B/Qwen2_vla-v0-robot-action-11_1_all_lora_film_w_reasoning/checkpoint-45000",
        # "model_path": "/media/eai/MAD-1/wjj/dit_head_qwen2_vla/2B/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_film_w_pretrain_dit/checkpoint-45000", # DiT_L  only pretrain dit
        "model_path": "/media/eai/MAD-1/wjj/dit_head_qwen2_vla/2B/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_film_w_pretrain_DiTL_ema/checkpoint-45000",# DiT_L  only pretrain dit
        # "model_path": "/media/eai/MAD-1/wjj/dit_head_qwen2_vla/2B/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_film_w_pretrain_DiTL_ema_gt_reasoning_all/checkpoint-45000",
        # "model_path": "/media/eai/MAD-1/wjj/dit_head_qwen2_vla/2B/Qwen2_vla-v0-robot-action-11_1_all_lora_film_w_reasoning_DiTL/checkpoint-45000", # DiT_L no pretrain dit
        # "model_base": f"/media/eai/WJJ1T/droid/results/llava_pythia/pythia_{model_size}/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune",
        # "model_path": "/media/eai/MAD-1/wjj/7B/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_film_residual/checkpoint-45000",
        # "model_path":f"/media/eai/ExtremePro/ljm/multi_head_qwen2/tiny_vla/qwen_tinyvla/checkpoint-80000",
        "model_base": f"/home/eai/Downloads/Qwen2-VL-{model_size}-Instruct",
        # "model_base": "/home/eai/wjj/72B_weights/Qwen2-VL-72B-Instruct",
        # "model_base": "/media/eai/PSSD-6/wjj/results/pythia_1B/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune",
        # 'pretrain_path': '/media/eai/PSSD-6/wjj/results/multi_head/Qwen2_vla-v0-robot-action-38k_droid_pretrain_all_reasoning_data_lora_all_w_reasoning/checkpoint-56000',
        # 'pretrain_path': '/media/eai/SanDisk/wjj/7B/Qwen2_vla-v0-robot-action-38kplus1k_droid_pretrain_w_reasoning_2e-5/checkpoint-80000',
        # "pretrain_path": '/media/eai/SanDisk/wjj/2B/Qwen2_vla-v0-robot-action-38k_droid_pretrain_lora_all_wo_film/checkpoint-40000',
        # "pretrain_path": "/media/eai/ExtremePro/wjj/Qwen2_vla-v0-robot-action-38k_droid_pretrain_lora_all_w_reasoning/checkpoint-200000",
        "pretrain_path": None,
        "enable_lora": True,
        "conv_mode": "pythia",
        "temp_agg": False,
        "action_head": action_head,
        'model_size': model_size,
        'save_model': False,
        "tinyvla": False,
    }

    global im_size
    im_size = 480  # default 480
    select_one = False  # select one embedding or using all
    raw_lang = 'I am hungry, is there anything I can eat?'
    # raw_lang = 'I want to paste a poster, can you help me?'
    # raw_lang = 'I want a container to put water in, can you help me?'

    raw_lang = 'Upright the tipped-over pot.'

    # raw_lang = 'Put the cup on the tea table and pour tea into the cup'

    # raw_lang = 'Put the white car into the drawer.'
    # raw_lang = "Solve the equation on the table."

    # raw_lang = "Arrange the objects according to their types."
    raw_lang = 'Classifying all objects and place to corresponding positions.'

    # raw_lang = "put the purple cube into the blue box."
    # raw_lang = "put the purple cube into the yellow box."
    # raw_lang = 'Put the cup onto the plate.'

    ### OOD Instruction
    # raw_lang = "Move any object on the right panel to the left basket."
    # raw_lang = "What is the object on the right panel?"

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    policy = None
    policy = qwen2_vla_policy(policy_config)

    eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=raw_lang,
            select_one=select_one)

    print()
    exit()

# [0.5553438067436218, 0.0022895748261362314, 0.6198290586471558, -3.119706407105779, -0.006210746497147035, -0.025821790776125078]

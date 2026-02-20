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


def get_obs(deplot_env_obs, stats, time=0, camera_views=4):
    cur_traj_data = dict()
    # (480, 270, 4)

    cur_bottom_rgb = deplot_env_obs['images']['cam_bottom']  # camera_extrinsics image
    cur_top_rgb = deplot_env_obs['images']['cam_top']  # camera_extrinsics image
    cur_left_rgb = deplot_env_obs['images']['cam_left_wrist']  # camera_extrinsics image
    cur_right_rgb = deplot_env_obs['images']['cam_right_wrist']  # camera_extrinsics image

    cur_bottom_rgb = cv2.resize(cv2.cvtColor(cur_bottom_rgb, cv2.COLOR_BGRA2BGR), (320, 240))[:, :, ::-1]
    cur_top_rgb = cv2.resize(cv2.cvtColor(cur_top_rgb, cv2.COLOR_BGRA2BGR), (320, 240))[:, :, ::-1]
    cur_left_rgb = cv2.resize(cv2.cvtColor(cur_left_rgb, cv2.COLOR_BGRA2BGR), (320, 240))[:, :, ::-1]
    cur_right_rgb = cv2.resize(cv2.cvtColor(cur_right_rgb, cv2.COLOR_BGRA2BGR), (320, 240))[:, :, ::-1]

    # cv2.imshow('cur_rgb', cv2.hconcat([cur_left_rgb, cur_right_rgb, cur_bottom_rgb, cur_top_rgb]))
    # cv2.waitKey(1)

    cur_right_depth = np.zeros_like(cur_right_rgb) - 1.0
    cur_right_depth = cur_right_depth[..., :1]
    cur_left_depth = np.zeros_like(cur_left_rgb) - 1.0
    cur_left_depth = cur_left_depth[..., :1]

    cur_joint_positions = deplot_env_obs['qpos']

    cur_state_np = pre_process(cur_joint_positions, 'qpos', stats)

    # [128, 128, 3] np array
    right_rgb_img = cur_right_rgb  # deplot_env_obs['front']
    right_depth_img = cur_right_depth
    left_rgb_img = cur_left_rgb  # deplot_env_obs['wrist_1']
    left_depth_img = cur_left_depth
    # cur_high_rgb = cur_top_rgb

    cur_state = cur_state_np  # deplot_env_obs['state']
    cur_state = np.expand_dims(cur_state, axis=0)

    # [2, 1, 128, 128, 3]
    # [2, 480, 480, 3]
    if camera_views == 4:
        traj_rgb_np = np.array([cur_bottom_rgb, cur_top_rgb, left_rgb_img, right_rgb_img])
    else:
        traj_rgb_np = np.array([cur_top_rgb, left_rgb_img, right_rgb_img])


    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))

    traj_depth_np = np.array([right_depth_img, left_depth_img])
    traj_depth_np = np.expand_dims(traj_depth_np, axis=1)
    traj_depth_np = np.transpose(traj_depth_np, (1, 0, 4, 2, 3))

    print("#" * 50)
    print(traj_rgb_np.shape)
    # traj_rgb_np = np.array([[cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB) for img in traj_rgb_np[0]]])
    # traj_rgb_np = np.transpose(traj_rgb_np, (0, 1, 4, 2, 3))
    return cur_joint_positions, cur_state, traj_rgb_np, traj_depth_np


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


def eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=None, select_one=False):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)

    rand_crop_resize = True
    model_config = policy.config.policy_head_config

    temporal_agg = policy_config['temp_agg']
    action_dim = model_config['input_dim']
    state_dim = model_config['state_dim']

    policy.policy.eval()

    import pickle
    paths = policy_config['model_path'].split('/')[:-1]
    if 'checkpoint' in paths[-1]:
        paths = paths[:-1]
    stats_path = os.path.join("/".join(paths), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    if 'fold_shirt' in stats.keys():
        if 'fold' in raw_lang.lower():
            stats = stats['fold_shirt']
        elif 'tablewares' in raw_lang.lower():
            stats = stats['clean_table']
        else:
            stats = stats['other']

    if policy_config["action_head"].lower() == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif 'diffusion' in policy_config["action_head"] or 'vqbet' in policy_config["action_head"]:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    env = deploy_env

    query_frequency = 25

    if temporal_agg:
        query_frequency = 1
        num_queries = int(query_frequency)
    else:
        query_frequency = int(query_frequency)
        num_queries = query_frequency
        from collections import deque
        action_queue = deque(maxlen=num_queries)

    max_timesteps = int(1000 * 10)  # may increase for real-world tasks
    temp = copy.deepcopy(query_frequency)

    for rollout_id in range(1000):

        rollout_id += 0

        # env.reset(randomize=False)

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
        time_cur = -1
        time_pre = -1
        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            for t in range(max_timesteps):
                if t < 10:
                    query_frequency = 16
                else:
                    query_frequency = 16

                time1 = time.time()

                obs = deploy_env.get_obs()

                cur_state_np_raw, robot_state, traj_rgb_np, traj_depth_np = get_obs(obs, stats, time=t,
                                                                                    camera_views=policy_config[
                                                                                        'camera_views'])
                # if t % 100 == 5:
                #     a = input("q means next eval:")
                #     if a== 'q':
                #         deploy_env.step('reset', mode=policy_config['control_mode'])
                #         lang_in = input("Input the raw_lang(q and enter mean using default):")
                #         if lang_in != 'q' or lang_in != '':
                #             raw_lang = lang_in
                #             print(raw_lang)
                #
                #         break

                # image_list.append(traj_rgb_np)
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

                image_list.append(curr_image)
                # control_timestamps["policy_start"] = time_ms()
                if t == 0:
                    # warm up
                    for _ in range(2):
                        batch = policy.process_batch_to_qwen2_vla(image_list, robot_state, raw_lang)
                        if policy_config['tinyvla']:
                            policy.policy.evaluate_tinyvla(**batch, is_eval=True, select_one=select_one,
                                                           tokenizer=policy.tokenizer)
                        else:
                            all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, select_one=select_one,
                                                                          tokenizer=policy.tokenizer)
                            print("*" * 50)
                            print(outputs)
                    print('network warm up done')
                    time1 = time.time()

                if t % query_frequency == 0:
                    process_time1 = time.time()
                    batch = policy.process_batch_to_qwen2_vla(image_list, robot_state, raw_lang)

                    if policy_config['tinyvla']:
                        all_actions, outputs = policy.policy.evaluate_tinyvla(**batch, is_eval=True,
                                                                              select_one=select_one,
                                                                              tokenizer=policy.tokenizer)
                    else:
                        all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, select_one=select_one,
                                                                      tokenizer=policy.tokenizer)
                    if not temporal_agg:
                        while len(action_queue) > 0:
                            action_queue.popleft()
                        action_queue.extend(
                            torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:num_queries])
                    process_time2 = time.time()

                    process_t = process_time2 - process_time1
                    print(
                        f"{RED} Execute >>{query_frequency}<< action costs {time_cur - time_pre - process_t}s. Model forward takes {process_t}s {RESET}")
                    time_pre = time_cur
                    time_cur = time.time()

                if temporal_agg:
                    # print(f"all_actions: {all_actions.size()}")
                    # print(f"all_time_actions: {all_time_actions.size()}")
                    # print(f"t: {t}, num_queries:{num_queries}")
                    # all_time_actions[[t], t:t + num_queries] = all_actions[:, :num_queries, :]
                    # actions_for_curr_step = all_time_actions[:, t]
                    # actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    # actions_for_curr_step = actions_for_curr_step[actions_populated]
                    # k = 0.01
                    # exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    # exp_weights = exp_weights / exp_weights.sum()
                    # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    # raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    raw_action = torch.zeros((14)).to('cuda')
                    raw_action[9] = 0.003
                    outputs = ''
                else:
                    raw_action = action_queue.popleft()

                # print(f"raw action size: {raw_action.size()}")
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                action = post_process(raw_action)
                print(f"after post_process action size: {action.shape}")
                # target_qpos = action

                # action = convert_actions(action.squeeze())
                print(f'step {t}, pred action: {outputs}{action}')
                if len(action.shape) == 2:
                    action = action[0]
                # action[7:] = 0
                action_info = deploy_env.step(action.tolist(), mode=policy_config['control_mode'])

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            # plt.close()

    return


if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    sys.path.insert(0, "/home/eai/Dev-Code/mirocs")
    from run.agilex_robot_env import AgilexRobot

    action_head = 'dit_diffusion_policy'  # 'unet_diffusion_policy'
    model_size = '2B'
    policy_config = {
        # ema
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_folding_shirt_lora_ema_finetune_dit_h_3wsteps/checkpoint-30000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_folding_shirt_lora_ema_finetune_dit_h_2/checkpoint-10000",
        # "model_path": "/home/eai/Documents/wjj/results/qwen2_vl_only_folding_shirt_lora_ema_finetune_dit_h_4w_steps/checkpoint-30000",

        # two stage - finetune
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_lora_combine_pretrain_DIT_H_align_finetune_2/checkpoint-10000",
        # "model_path": "/home/eai/Documents/wjj/results/qwen2_vl_only_fold_shirt_lora_combine_substep_pretrain_DIT_H_align_finetune_2w_steps/checkpoint-20000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_lora_combine_substep_pretrain_DIT_H_align_finetune_2w_steps_EMA_norm_stats/checkpoint-20000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_lora_combine_substep_pretrain_DIT_H_align_finetune_2w_steps_freeze_VLM_EMA/checkpoint-20000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_lora_combine_substep_pretrain_DIT_H_align_finetune_2w_steps_norm_stats2_chunk_50/checkpoint-20000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_lora_combine_pretrain_DIT_H_align_finetune_2w_steps_norm_stats2_chunk_50_correct_1w_steps/checkpoint-10000",

        # two stage - align
        # "model_path": "/home/eai/Documents/wjj/results/qwen2_vl_all_data_1200_align_frozen_dit_lora_substep/checkpoint-40000",

        # full parameter training
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_combine_pretrain_DIT_H_full_param/checkpoint-40000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_4_cameras_all_data_1_12_pretrain_DIT_H_full_param_pretrain/checkpoint-60000",

        # "model_path": "/media/eai/MAD-2/wjj/qwen2_vl_4_cameras_1_12_all_data_pretrain_DiT_XH_full_param_stage_1_50/checkpoi nt-60000", #2B
        # "model_path": "/media/eai/MAD-2/wjj/qwen2_vl_4_cameras_all_data_1_12_pretrain_DIT_H_full_param_pretrain/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_1_17_all_data_pretrain_DiT_H_full_param_stage_1_50/checkpoint-60000",
        "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_1_12_all_data_pretrain_DiT_H_full_param_stage_1_50/checkpoint-60000",
        "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_1_17_all_data_pretrain_4w_DiT_H_full_param_stage_1_50/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_1_17_all_data_pretrain_6w_DiT_H_Non_EMA_full_param_stage_1_50/checkpoint-60000", # Non EMA DiT aa11

        "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/qwen2_vl_3_cameras_1_17_all_data_pretrain_6w_DiT_H_Non_EMA_full_param_stage_1_50/checkpoint-60000", # stage 2 best for standard folding shirt

        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_all_data_1_17_3_cameras_1_17_all_data_pretrain_6w_DiT_H_Non_EMA_full_param_stage_1_50_12w/checkpoint-30000",
        # best for standard folding shirt
        # "model_path": "/home/eai/wjj/ckpts/qwen2_vl_3_cameras_all_data_1_17_3_cameras_1_17_all_data_pretrain_6w_DiT_H_Non_EMA_full_param_stage_1_50_12w/checkpoint-30000",
        # best for standard folding shirt

        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_all_data_1_23_pretrain_5w_DiT_H_1_23_full_param_stage_1_50/checkpoint-100000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_all_data_1_25_multi_embodiment_DiT_Non_EMA_H_1_25_full_param_stage_1_50/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_4_cameras_1_17_all_data_pretrain_4w_DiT_H_1_17_full_param_stage_1_50_raw_lang/checkpoint-60000", # non substeps
        # post training
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_combine_pretrain_DIT_H_full_param_post_training/checkpoint-20000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_combine_pretrain_DIT_H_full_param_post_training_6w/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_combine_pretrain_DIT_H_full_param_post_training_constant_lr/checkpoint-60000", # constant lr
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_combine_pretrain_814_DIT_H_full_param_post_training_814_trajs_16/checkpoint-20000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_combine_constant_pretrain_DIT_H_full_param_post_training_814_trajs_16/checkpoint-20000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_1_4_combine_constant_pretrain_DIT_H_full_param_post_training_711_trajs_16_2w/checkpoint-20000", # constant pretrain dit
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_3_cameras_fold_shirt_1_17_combine_constant_pretrain_DIT_H_full_param_post_training_50_4w/checkpoint-20000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_only_fold_shirt_1_19_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_2w/checkpoint-20000", # aa11
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_only_fold_shirt_1_19_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_6w/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/GRPO_qwen2_vl_3_cameras_random_folding_1_25_combine_pretrain_Non_EMA_DIT_H_full_param_post_training_50_6w/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_only_unloading_dryer_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_1w/checkpoint-10000",

        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_standard_folding_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_3w/checkpoint-30000",  # best for standard folding shirt
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_1_12_combine_constant_pretrain_DIT_H_full_param_post_training_50_2w/checkpoint-20000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_23_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_6w/checkpoint-60000",
        # best one for random
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_aloha_folding_shirt_lerobot_1_25_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_6w/checkpoint-60000",

        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_25_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_6w/checkpoint-80000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_25_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_6w/checkpoint-80000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_25_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_6w/checkpoint-60000",
        # "model_path": "/media/eai/MAD-2/wjj/qwen2_vl_3_cameras_random_folding_1_25_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_6w/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_25_combine_constant_pretrain_Non_EMA_DIT_H_9w_full_param_post_training_50_6w_2/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_high_quaility_combine_constant_pretrain_Non_EMA_DIT_H_9w_full_param_post_training_50_6w_2/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_25_combine_constant_pretrain_Non_EMA_DIT_H_10w_full_param_post_training_50_6w/checkpoint-60000", # non constant(name error)
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_25_1_17_6w_DiT_Non_EMA_post_training_stage_2_50/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_25_stage3_0117_stage2_0117_stage1_50/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_23_stage3_0117_stage2_0117_stage1_50_first_layer_input_embedding/checkpoint-60000",

        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_25_multi_embodiment_DiT_Non_EMA_H_1_25_post_training_stage_2_50/checkpoint-60000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/lerobot_qwen2_vl_folding_blue_shirt_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_2w/checkpoint-20000",
        # tinyvla

        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_all_data_1200_pretrain_DiT_H_tinyvla/checkpoint-40000",

        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_all_data_1_17_stage2_0117_stage1_50_without_film/checkpoint-120000", # without film
        # "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/qwen2_vl_aloha_all_1_17_combine_constant_pretrain_Non_EMA_DIT_H_full_param_wo_film2/checkpoint-100000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_aloha_all_1_17_combine_constant_pretrain_Non_EMA_DIT_H_full_param_encode_state2/checkpoint-100000", #with state embedding
        # "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/qwen2_vl_aloha_all_1_17_combine_constant_pretrain_Non_EMA_DIT_H_full_param_encode_state3/checkpoint-80000", #with state embedding
        # "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/qwen2_vl_aloha_all_1_17_combine_constant_pretrain_Non_EMA_DIT_H_full_param_encode_state_after_vision/checkpoint-100000", #with state embedding insert middle
        
        # "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/folding_two_shirts_by_drag_stage3_DiT_H/checkpoint-40000", # fold two

        # "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/aloha_all_1_17_Stage2_DIT_H_Stage1_1_17_no_film/checkpoint-100000", # no film

        # "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/folding_two_shirts_by_drag_stage3_DiT_H_long/checkpoint-100000", # drag cloths

        # "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/aloha_all_1_17_Stage2_DIT_H_Stage1_1_17_using_state_correct/checkpoint-40000", # using_state

        # paligemma
        # "model_path": "/media/eai/MAD-1/wjj/paligemma_3b_aloha/paligemma_aloha_all_1_17_combine_constant_pretrain_Non_EMA_DIT_H_full_param/checkpoint-100000",
        # from scratch DiT + VLM
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_folding_shirt_lora_ema_scratch_dit_h/checkpoint-80000",
        # paligemma
        # "model_path": "/home/eai/Documents/wjj/evaluate/aloha_results/paligemma_3B/paligemma-v0-robot-action-aloha_clean_table_folding_shirt_tinyvla_lora2/checkpoint-40000",
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl-v0-robot-action-clean_table_fold_shirt_pretrain_dit_lora_only_folding_shirt/checkpoint-5000",
        # "model_path": "/media/eai/MAD-1/wjj/paligemma_3b_aloha/paligemma-v0-robot-action-clean_table_fold_shirt_pretrain_dit_lora/checkpoint-60000",

        # "model_base": f"/home/eai
        # /Downloads/Qwen2-VL-{model_size}-Instruct",
        # "model_base": "/home/eai/Documents/wjj/evaluate/vla-paligemma-3b-pt-224",
        "model_base": None,
        # "pretrain_dit_path": f"/home/eai/Documents/ljm/scaledp/filmresnet50_with_lang_sub_reason/fold_t_shirt_easy_version_1212_DiT-L_320_240_32_1e-4_numsteps_100000_scaledp_429traj_12_16/policy_step_100000.ckpt",
        "pretrain_dit_path": None,
        # "pretrain_path": '/media/eai/PSSD-6/wjj/results/aloha/Qwen2_vla-v0-robot-action-38k_droid_pretrain_lora_all_wo_film/checkpoint-40000',
        # "pretrain_path": "/home/eai/Documents/wjj/results/qwen2_vl_all_data_1200_align_frozen_dit_lora_substep/checkpoint-40000",
        # "pretrain_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_all_data_1200_align_frozen_dit_lora_substep_chunk_50/checkpoint-40000",
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
    # raw_lang ='The crumpled shirts are in the basket. Pick it and fold it.'

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    policy = None
    agilex_bot = AgilexRobot()
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

    eval_bc(policy, agilex_bot, policy_config, save_episode=True, num_rollouts=1, raw_lang=raw_lang,
            select_one=select_one)

    print()
    exit()


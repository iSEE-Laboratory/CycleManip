import os
import torch
import cv2
import time
import sys
import pickle
import numpy as np
import torch_utils as TorchUtils

from torchvision import transforms

from vla import *
from policy_heads import *

from aloha_scripts.constants import *
from data_utils.dataset import set_seed
from data_utils.robot_data_processor import InternVL3Process
from vla.model_load_utils import load_model_for_eval


def init_robot():
    sys.path.insert(0, "/home/eai/Dev-Code/droid_ori")
    from droid.robot_env import RobotEnv

    policy_timestep_filtering_kwargs = {'action_space': 'cartesian_position', 'gripper_action_space': 'position',
                                        'robot_state_keys': ['cartesian_position', 'gripper_position',
                                                             'joint_positions']}
    # resolution (w, h)
    policy_camera_kwargs = {
        'hand_camera': {'image': True, 'concatenate_images': False, 'resolution': (640, 480), 'resize_func': 'cv2'},
        'varied_camera': {'image': True, 'concatenate_images': False, 'resolution': (640, 480), 'resize_func': 'cv2'}}

    deploy_env = RobotEnv(
        action_space=policy_timestep_filtering_kwargs["action_space"],
        gripper_action_space=policy_timestep_filtering_kwargs["gripper_action_space"],
        camera_kwargs=policy_camera_kwargs
    )
    deploy_env._robot.establish_connection()
    deploy_env.camera_reader.set_trajectory_mode()
    return deploy_env


def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp


def preprocess_img(images: torch.Tensor):
    assert images.ndim == 4 and images.shape[1] == 3
    original_size = (480, 640)
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


def get_obs(deplot_env_obs, stats):
    # >>>>>>>>>>>>>>>>> image resize <<<<<<<<<<<<<<<<<
    cur_right_rgb = deplot_env_obs['image']['23343100_left']  # camera_extrinsics image
    cur_left_rgb = deplot_env_obs['image']['23282896_left']  # camera_extrinsics image
    cur_wrist_rgb = deplot_env_obs['image']['18361939_left']  # camera_extrinsics image
    cur_wrist_rgb = cv2.resize(cur_wrist_rgb, (640, 480))

    w, h = 640, 480
    center = (w // 2, h // 2)
    angle = 180
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    cur_wrist_rgb = cv2.warpAffine(cur_wrist_rgb, M, (w, h))

    cur_right_rgb = cv2.cvtColor(cur_right_rgb, cv2.COLOR_BGRA2BGR)[:, :, ::-1]
    cur_left_rgb = cv2.cvtColor(cur_left_rgb, cv2.COLOR_BGRA2BGR)[:, :, ::-1]
    cur_wrist_rgb = cv2.cvtColor(cur_wrist_rgb, cv2.COLOR_BGRA2BGR)[:, :, ::-1]

    # >>>>>>>>>>>>>>>>> state <<<<<<<<<<<<<<<<<
    cur_cartesian_position = np.array(deplot_env_obs['robot_state']['cartesian_position'])
    cur_gripper_position = np.expand_dims(np.array(deplot_env_obs['robot_state']['gripper_position']), axis=0)
    cur_state_np_raw = np.concatenate((cur_cartesian_position, cur_gripper_position))
    cur_state_np = pre_process(cur_state_np_raw, 'qpos', stats)
    cur_state = cur_state_np
    cur_state = np.expand_dims(cur_state, axis=0)

    # >>>>>>>>>>>>>>>>> image crop and resize, similar to the train image preprocess <<<<<<<<<<<<<<<<<
    cur_left_rgb = np.array(cur_left_rgb)
    cur_right_rgb = np.array(cur_right_rgb)
    cur_wrist_rgb = np.array(cur_wrist_rgb)
    curr_images = np.array([cur_left_rgb, cur_right_rgb, cur_wrist_rgb])
    curr_images = np.transpose(curr_images, (0, 3, 1, 2))
    curr_images = torch.from_numpy(curr_images)

    # >>>>>>>>>>>>>>>>> image preprocess <<<<<<<<<<<<<<<<<
    traj_rgb = preprocess_img(curr_images)

    return cur_state_np_raw, cur_state, traj_rgb


def convert_actions(pred_action):
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)

    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
    print(f'4. after convert pred_action: {pred_action}')

    return pred_action


class vla_policy:
    def __init__(self, policy_config, camera_names):
        super(vla_policy).__init__()
        self.camera_names = camera_names
        self.load_policy(policy_config)

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config['enable_lora'] else None
        model_path = policy_config["model_path"]
        self.tokenizer, self.policy = load_model_for_eval(
            model_path=model_path,
            model_base=model_base,
            policy_config=policy_config)

        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        self.vla_process = InternVL3Process(
            tokenizer=self.tokenizer,
            conv_template=self.policy.conv_template,
            camera_names=self.camera_names,
            num_image_token=self.policy.num_image_token
        )

    def precess_input(self, sample):
        data_dict = self.vla_process.preprocess(sample)
        return data_dict


def eval_bc(policy, env, policy_config, raw_lang=None):
    assert raw_lang is not None
    set_seed(0)

    rand_crop_resize = True
    model_config = policy.config.policy_head_config

    action_dim = getattr(model_config, 'input_dim', 10)
    state_dim = getattr(model_config, 'state_dim', 7)

    policy.policy.eval()

    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    query_frequency = 16 // 1
    num_queries = query_frequency
    from collections import deque
    action_queue = deque(maxlen=num_queries)

    max_timesteps = int(1000 * 10)

    for rollout_id in range(1000):
        rollout_id += 0
        env.reset(randomize=False)
        print(f"env has reset!")

        with torch.inference_mode():
            DT = 1 / FPS
            for t in range(max_timesteps):
                if t % 100 == 1:
                    a = input("q means next eval:")
                    if a == 'q':
                        env.reset(randomize=False)
                        action_queue = deque(maxlen=num_queries)
                        lang_in = input("Input the raw_lang(q means using default lang):")
                        if lang_in != 'q' or lang_in != '':
                            raw_lang = lang_in
                            print(raw_lang)
                        break

                obs = env.get_observation()
                cur_state_np_raw, robot_state, traj_rgb = get_obs(obs, stats)
                robot_state = torch.from_numpy(robot_state).float().cuda()
                curr_image = traj_rgb.cuda()
                sample = {
                    "image": curr_image,
                    "raw_lang": raw_lang,
                    "state": robot_state
                }

                if t == 0:
                    for _ in range(2):
                        batch = policy.precess_input(sample)
                        all_actions = policy.policy.sample_action(**batch)
                    print('network warm up done')

                if len(action_queue) == 0:
                    batch = policy.precess_input(sample)
                    all_actions = policy.policy.sample_action(**batch)
                    action_queue.extend(
                        torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:num_queries])

                raw_action = action_queue.popleft()

                print(f"raw action size: {raw_action.size()}")
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                action = post_process(raw_action)
                print(f"step {t}, after post_process action size: {action.shape}")

                action = convert_actions(action.squeeze())
                _ = deploy_env.step(action)

    return


if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> hyper parameters <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'unet_diffusion_policy'
    task_name = "mobile_franka_bin_picking"
    task_config = TASK_CONFIGS[task_name]
    camera_names = task_config['camera_names']
    BS = 128
    LR = "2e-5"
    noise_samples = 8
    ckpt_name = "checkpoint-20000"
    model_dir = (f"/media/eai/Elements/robotics/model_Param/mobile_franka_param/tinyvla/unet_diffusion_policy_results/"
                 f"{task_name}-{BS}BS-{LR}LR-{noise_samples}noise_samples/{ckpt_name}")

    policy_config = {
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Full Parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        "model_path": model_dir,
        "model_base": f"/home/eai/zhumj/mllm_param/InternVL3-1B",
        "enable_lora": False,
        "action_head": action_head,
    }

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> init policy <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    policy = vla_policy(policy_config, camera_names)

    # raw_lang = "Move the tennis ball on the right panel into the left box."
    # raw_lang = "Move the cutter knife on the right panel into the left box."
    raw_lang = "Move objects on the table to the box in the following order: mug, toy pig and tennis ball."

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> init robot <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    deploy_env = init_robot()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> eval bc <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    eval_bc(policy, deploy_env, policy_config, raw_lang=raw_lang)

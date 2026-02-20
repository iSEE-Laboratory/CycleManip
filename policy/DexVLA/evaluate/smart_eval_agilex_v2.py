import os.path

from torchvision import transforms
from aloha_scripts.utils import *
import time
from data_utils.dataset import set_seed
from einops import rearrange

import sys
from policy_heads import *
from dex_vla.utils.image_processing_qwen2_vla import *
from paligemma_vla.utils.processing_paligemma_vla import *
from dex_vla.utils.processing_qwen2_vla import *
from vla_policy import *

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
    cur_bottom_rgb = deplot_env_obs['images']['cam_bottom']
    cur_top_rgb = deplot_env_obs['images']['cam_top']
    cur_left_rgb = deplot_env_obs['images']['cam_left_wrist']
    cur_right_rgb = deplot_env_obs['images']['cam_right_wrist']

    cur_bottom_rgb = cv2.cvtColor(cur_bottom_rgb, cv2.COLOR_BGRA2BGR)[:, :, ::-1]
    cur_top_rgb = cv2.cvtColor(cur_top_rgb, cv2.COLOR_BGRA2BGR)[:, :, ::-1]
    cur_left_rgb = cv2.cvtColor(cur_left_rgb, cv2.COLOR_BGRA2BGR)[:, :, ::-1]
    cur_right_rgb = cv2.cvtColor(cur_right_rgb, cv2.COLOR_BGRA2BGR)[:, :, ::-1]

    cur_joint_positions = deplot_env_obs['qpos']

    cur_state_np = pre_process(cur_joint_positions, 'qpos', stats)

    cur_state = cur_state_np  # deplot_env_obs['state']
    cur_state = np.expand_dims(cur_state, axis=0)

    # [2, 1, 128, 128, 3]
    # [2, 480, 480, 3]
    if camera_views == 4:
        traj_rgb_np = np.array([cur_bottom_rgb, cur_top_rgb, cur_left_rgb, cur_right_rgb])
    else:
        traj_rgb_np = np.array([cur_top_rgb, cur_left_rgb, cur_right_rgb])

    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))

    print("#" * 50)
    print(traj_rgb_np.shape)

    return cur_joint_positions, cur_state, traj_rgb_np


def eval_bc(policy, deploy_env, policy_config, raw_lang=None, query_frequency=25):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)

    rand_crop_resize = True
    model_config = policy.config.policy_head_config

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

    action_queue = deque(maxlen=query_frequency)

    max_timesteps = int(1000 * 10)  # may increase for real-world tasks
    time_cur = -1
    time_pre = -1
    for rollout_id in range(1000):

        rollout_id += 0

        print(f"env has reset!")
        robot_state_history = np.zeros((max_timesteps, state_dim))
        image_list = []  # for visualization

        with torch.inference_mode():
            time0 = time.time()
            for t in range(max_timesteps):

                time1 = time.time()
                obs = deploy_env.get_obs()
                cur_state_np_raw, robot_state, traj_rgb_np = get_obs(obs, stats, time=t, camera_views=policy_config['camera_views'])
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

                robot_state_history[t] = cur_state_np_raw
                robot_state = torch.from_numpy(robot_state).float().cuda()
                curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
                if rand_crop_resize:
                    print('rand crop resize is used!')
                    original_size = curr_image.shape[-2:]
                    ratio = 0.95
                    curr_image = curr_image[...,
                                 int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                 int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                    curr_image = curr_image.squeeze(0)
                    resize_transform = transforms.Resize((240, 320), antialias=True)
                    curr_image = resize_transform(curr_image)
                    curr_image = curr_image.unsqueeze(0)

                image_list.append(curr_image)

                if t % query_frequency == 0:
                    process_time1 = time.time()
                    batch = policy.process_batch_to_qwen2_vla(image_list, robot_state, raw_lang)

                    if policy_config['tinyvla']:
                        all_actions, outputs = policy.policy.evaluate_tinyvla(**batch, is_eval=True, tokenizer=policy.tokenizer)
                    else:
                        all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, tokenizer=policy.tokenizer, raw_images=curr_image)

                    while len(action_queue) > 0:
                        action_queue.popleft()
                    action_queue.extend(
                            torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:query_frequency])

                    process_time2 = time.time()
                    process_t = process_time2 - process_time1
                    print(
                        f"{RED} Execute >>{query_frequency}<< action costs {time_cur - time_pre - process_t}s. Model forward takes {process_t}s {RESET}")
                    time_pre = time_cur
                    time_cur = time.time()

                raw_action = action_queue.popleft()

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                action = post_process(raw_action)
                print(f"after post_process action size: {action.shape}")

                print(f'step {t}, pred action: {outputs}{action}')
                if len(action.shape) == 2:
                    action = action[0]
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

        # Stage 2
        "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/qwen2_vl_3_cameras_1_17_all_data_pretrain_6w_DiT_H_Non_EMA_full_param_stage_1_50/checkpoint-60000", # stage 2 best for standard folding shirt
        # "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/aloha_all_1_17_Stage2_DIT_H_Stage1_1_17_using_state_correct/checkpoint-60000", # using_state
        "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/aloha_all_1_17_Stage2_DIT_H_Stage1_1_17_standard/checkpoint-40000",

        # "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/aloha_all_1_17_Stage2_DIT_H_Stage1_1_17_wo_film_correct/checkpoint-60000", # wo film
        # "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/aloha_all_1_17_Stage2_DIT_H_Stage1_1_17_external_resnet/checkpoint-60000", # external resnet
        # Stage 3
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_25_stage3_0117_stage2_0117_stage1_50/checkpoint-60000", # data ablate random folding
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_23_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_6w/checkpoint-60000", # best one for random
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_standard_folding_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_3w/checkpoint-30000",  # best for standard folding shirt
        # "model_path": "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_3_cameras_random_folding_1_25_combine_constant_pretrain_Non_EMA_DIT_H_full_param_post_training_50_6w/checkpoint-130000",
        # "model_path": "/media/eai/MAD-1/wjj/lerobot_qwen2_vla_aloha/folding_two_shirts_by_drag_stage3_DiT_H_long/checkpoint-100000", # drag cloths

        "model_base": None,
        "pretrain_dit_path": None,
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
    if not os.path.exists(os.path.join(policy_config['model_path'], "chat_template.json")):
        raise "Checkpoint must have chat_template.json and preprocessor.json"
    query_frequency = 8
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

    eval_bc(policy, agilex_bot, policy_config, raw_lang=raw_lang,
            query_frequency=query_frequency)

    print()
    exit()


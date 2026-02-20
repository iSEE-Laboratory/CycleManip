import sys
import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import yaml, json


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # 加载state和action
        joint_state_vector = root["/joint_state/vector"][()]
        joint_action_vector = root["/joint_action/vector"][()]
        
        # 加载图像数据
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return joint_state_vector, joint_action_vector, image_dict


def images_encoding(imgs):
    """将图像编码为JPEG格式"""
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def get_task_config(task_name):
    with open(f"./task_config/{task_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args


def data_transform(path, episode_num, save_path):
    """
    将真实机器人数据转换为pi0格式
    
    Args:
        path: 数据源路径
        episode_num: 要处理的episode数量
        save_path: 保存路径
    """
    begin = 0
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        print(f"Processing episode: {i + 1} / {episode_num}", end="\r")

        # 加载instructions (如果存在)
        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        if os.path.exists(instruction_data_path):
            with open(instruction_data_path, "r") as f_instr:
                instruction_dict = json.load(f_instr)
            instructions = instruction_dict.get(desc_type, ["No instruction available"])
        else:
            instructions = ["No instruction available"]
        
        save_instructions_json = {"instructions": instructions}

        # 创建episode目录
        os.makedirs(os.path.join(save_path, f"episode_{i}"), exist_ok=True)

        # 保存instructions
        with open(
                os.path.join(os.path.join(save_path, f"episode_{i}"), "instructions.json"),
                "w",
        ) as f:
            json.dump(save_instructions_json, f, indent=2)

        # 加载数据
        state_all, action_all, image_dict = load_hdf5(
            os.path.join(path, "data", f"episode{i}.hdf5")
        )

        # 初始化数据列表
        qpos = []  # 状态
        actions = []  # 动作
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        
        # 只有右臂: [right_arm(7), right_gripper(1)]
        left_arm_dim_val = 0  # 没有左臂
        right_arm_dim_val = 7  # 右臂7个关节
        
        left_arm_dim = []
        right_arm_dim = []

        # 不需要用上一个state预测下一个state,有控制量
        # state, action, images 长度应该相同
        assert state_all.shape[0] == action_all.shape[0], \
            f"Data shape mismatch: state={state_all.shape[0]}, action={action_all.shape[0]}"

        for j in range(state_all.shape[0]):
            state = state_all[j]
            action = action_all[j]
            
            # 保存state (qpos)
            qpos.append(state.astype(np.float32))
            
            # 保存action
            actions.append(action.astype(np.float32))
            
            # 保存维度信息
            left_arm_dim.append(left_arm_dim_val)
            right_arm_dim.append(right_arm_dim_val)

            # 处理图像
            # 根据你的相机配置调整相机名称
            # head_camera, right_camera, left_camera
            if "head_camera" in image_dict:
                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)
            
            if "right_camera" in image_dict:
                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)
            
            if "left_camera" in image_dict:
                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

        # 保存为HDF5文件
        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            
            # 保存图像
            image = obs.create_group("images")
            if len(cam_high) > 0:
                cam_high_enc, len_high = images_encoding(cam_high)
                image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            
            if len(cam_right_wrist) > 0:
                cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
                image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            
            if len(cam_left_wrist) > 0:
                cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
                image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        begin += 1
        print(f"Process episode {i} success!                    ")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str, help="Task setting/config name")
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    load_dir = os.path.join("../../data", str(task_name), str(setting))

    print(f'Reading data from path: {load_dir}')

    target_dir = f"processed_data/{task_name}-{setting}-{expert_data_num}"
    
    begin = data_transform(
        load_dir,
        expert_data_num,
        target_dir,
    )
    
    print(f"\n✓ Successfully processed {begin} episodes!")
    print(f"✓ Data saved to: {target_dir}")

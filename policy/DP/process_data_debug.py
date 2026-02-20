import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py
import json


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        vector = root["/joint_action/vector"][()]
        pointcloud = root["/pointcloud"][()]

        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

        endpose = root["/endpose"][()]


    return left_gripper, left_arm, right_gripper, right_arm, vector, pointcloud, endpose, image_dict


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)

    total_count = 0

    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    point_cloud_arrays = []
    episode_ends_arrays, action_arrays, state_arrays, endpose_arrays, joint_action_arrays, loop_times_arrays, head_camera_arrays = (
        [], [], [], [], [], [], []
    )
    # 三种指令类型的数组
    instruction_int_arrays, instruction_sim_arrays, instruction_full_arrays = [], [], []

    # 物体姿态数组 - 将所有物体姿态拼接成一个数组
    objpose_arrays = []
    objpose_keys = []

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        # Load instruction data - 加载三种指令类型
        desc_type = "seen"


        instruction_sim_data_path = os.path.join(load_dir, "instructions_sim", f"episode{current_ep}.json")
        with open(instruction_sim_data_path, "r") as f_instr:
            instruction_sim_dict = json.load(f_instr)
        instructions_sim = instruction_sim_dict[desc_type]

        instruction_full_data_path = os.path.join(load_dir, "instructions_full", f"episode{current_ep}.json")
        with open(instruction_full_data_path, "r") as f_instr:
            instruction_full_dict = json.load(f_instr)
        instructions_full = instruction_full_dict[desc_type]

        try:
            instruction_int_data_path = os.path.join(load_dir, "instructions_int", f"episode{current_ep}.json")
            with open(instruction_int_data_path, "r") as f_instr:
                instruction_int_dict = json.load(f_instr)
            instructions_int = instruction_int_dict[desc_type]
        except:
            instructions_int = instructions_sim

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            pointcloud_all,
            endpose_all,
            image_dict_all,
        ) = load_hdf5(load_path)


        for j in range(0, left_gripper_all.shape[0]):
            
            head_img_bit = image_dict_all["head_camera"][j]
            pointcloud = pointcloud_all[j]
            joint_state = vector_all[j]
            endpose = endpose_all[j]

            if j != left_gripper_all.shape[0] - 1:
                head_img = cv2.imdecode(np.frombuffer(head_img_bit, np.uint8), cv2.IMREAD_COLOR)
                head_camera_arrays.append(head_img)
                point_cloud_arrays.append(pointcloud)
                state_arrays.append(joint_state)
                endpose_arrays.append(endpose)
                # 保存三种指令类型
                instruction_int_arrays.append(instructions_int)
                instruction_sim_arrays.append(instructions_sim)
                instruction_full_arrays.append(instructions_full)


            if j != 0:
                joint_action_arrays.append(joint_state)

        current_ep += 1
        total_count += left_gripper_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

    print()
    try:
        episode_ends_arrays = np.array(episode_ends_arrays)
        state_arrays = np.array(state_arrays)
        endpose_arrays = np.array(endpose_arrays)
        point_cloud_arrays = np.array(point_cloud_arrays)
        joint_action_arrays = np.array(joint_action_arrays)
        # 转换三种指令数组
        instruction_int_arrays = np.array(instruction_int_arrays)
        instruction_sim_arrays = np.array(instruction_sim_arrays)
        instruction_full_arrays = np.array(instruction_full_arrays)

        # 转换loop times和loop counter数组
        loop_times_arrays = np.array(loop_times_arrays)

        head_camera_arrays = np.array(head_camera_arrays)
        head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
    
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        joint_chunk_size = (100, joint_action_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1])
        head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
        instruction_chunk_size = (100,)
        loop_times_chunk_size = (100,)
        zarr_data.create_dataset(
            "head_camera",
            data=head_camera_arrays,
            chunks=head_camera_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "point_cloud",
            data=point_cloud_arrays,
            chunks=point_cloud_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "state",
            data=state_arrays,
            chunks=state_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "endpose",
            data=endpose_arrays,
            chunks=state_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "action",
            data=joint_action_arrays,
            chunks=joint_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        # 创建三种指令数据集
        zarr_data.create_dataset(
            "instruction_int",
            data=instruction_int_arrays,
            chunks=instruction_chunk_size,
            dtype=object,
            object_codec=zarr.codecs.JSON(),
            overwrite=True,
        )
        zarr_data.create_dataset(
            "instruction_sim",
            data=instruction_sim_arrays,
            chunks=instruction_chunk_size,
            dtype=object,
            object_codec=zarr.codecs.JSON(),
            overwrite=True,
        )
        zarr_data.create_dataset(
            "instruction_full",
            data=instruction_full_arrays,
            chunks=instruction_chunk_size,
            dtype=object,
            object_codec=zarr.codecs.JSON(),
            overwrite=True,
        )


        zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends_arrays,
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
    except ZeroDivisionError as e:
        print("If you get a `ZeroDivisionError: division by zero`, check that `data/pointcloud` in the task config is set to true.")
        raise 
    except Exception as e:
        print(f"An unexpected error occurred ({type(e).__name__}): {e}")
        raise

if __name__ == "__main__":
    main()

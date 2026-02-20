"""
Script to convert Aloha hdf5 data (real robot with single head camera) to the LeRobot dataset v2.0 format.

This version is specifically designed for real robot data with:
- Only head camera (cam_high)
- Only right arm (7 joints + 1 gripper)

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin_real.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # 只有右臂: 7个自由度 (包括夹爪)
    motors = [
        "right_joint_1",
        "right_joint_2",
        "right_joint_3",
        "right_joint_4",
        "right_joint_5",
        "right_joint_6",
        "right_gripper",
    ]

    # 只有头部相机
    cameras = [
        "cam_high",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors), ),  # 7 DoF
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    """加载相机图像数据"""
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                imgs_array.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
]:
    """加载单个episode的数据"""
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        # 只加载头部相机
        imgs_per_cam = load_raw_images_per_camera(
            ep,
            ["cam_high"],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    """填充数据集"""
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        
        # 加载instruction
        dir_path = os.path.dirname(ep_path)
        json_path = f"{dir_path}/instructions.json"

        if os.path.exists(json_path):
            with open(json_path, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
                instructions = instruction_dict['instructions']
                instruction = np.random.choice(instructions)
        else:
            instruction = "No instruction available"
        
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
            }

            # 只添加头部相机图像
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            
            dataset.add_frame(frame)
        
        dataset.save_episode()

    return dataset


def port_aloha_real(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """
    转换真实机器人数据到LeRobot格式
    
    Args:
        raw_dir: 原始数据目录
        repo_id: LeRobot数据集ID
        raw_repo_id: 原始数据集ID (可选)
        task: 任务名称
        episodes: 要处理的episode列表 (可选,默认处理所有)
        push_to_hub: 是否推送到Hugging Face Hub
        is_mobile: 是否是移动机器人
        mode: 图像模式 ("video" 或 "image")
        dataset_config: 数据集配置
    """
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")

    # 查找所有HDF5文件
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(Path(file_path))
    
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {raw_dir}")
    
    print(f"Found {len(hdf5_files)} HDF5 files")

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )

    if push_to_hub:
        dataset.push_to_hub()
    
    print(f"✓ Dataset created successfully at {HF_LEROBOT_HOME / repo_id}")


if __name__ == "__main__":
    tyro.cli(port_aloha_real)

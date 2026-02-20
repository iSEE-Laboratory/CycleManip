import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '..'))
sys.path.append(os.path.join(parent_directory, '../..'))

from typing import Dict
import torch
from torch.utils.data._utils.collate import default_collate

import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    get_val_mask,
    downsample_mask,
)
from diffusion_policy_3d.common.binary_sampler import BinarySequenceSampler
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import pdb
import clip


class BinaryRobotDataset(BaseDataset):

    def __init__(
        self,
        zarr_path,
        action_horizon=8,
        obs_horizon=8, 
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        use_language=True,  # Hard-coded to use language instruction
        sampler_strategy=None,
        agent_pose_type="state",
        action_sampling=None
    ):
        super().__init__()
        self.task_name = task_name
        self.use_language = use_language
        self.agent_pose_type = agent_pose_type
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        zarr_path = os.path.join(parent_directory, zarr_path)
        
        # Prepare keys to load from zarr
        keys_to_load = ["state", "action", "point_cloud"]
        self.language_key = ["instruction", "instruction_full", "instruction_int", "instruction_sim"]
        self.loop_key = ["loop_times", "loop_counter", "loop_length", "loop_curlen", "loop_now"]
        self.pose_key = ["endpose"]
        if use_language:
            keys_to_load.extend(self.language_key)
        keys_to_load.extend(self.loop_key)
        keys_to_load.extend(self.pose_key)

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys_to_load)
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        
        self.sampler = BinarySequenceSampler(
            replay_buffer=self.replay_buffer,
            action_horizon=action_horizon,
            obs_horizon=obs_horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            sampler_strategy=sampler_strategy,
            action_sampling=action_sampling,
        )
        self.train_mask = train_mask
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.sampler_strategy = sampler_strategy

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = BinarySequenceSampler(
            replay_buffer=self.replay_buffer,
            action_horizon=self.action_horizon,
            obs_horizon=self.obs_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            sampler_strategy=self.sampler_strategy,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "short_state": self.replay_buffer["state"][..., :],
        }
        if "point_cloud" in self.replay_buffer:
            data["point_cloud"] = self.replay_buffer["point_cloud"]
        
        # Note: instruction is not added to normalizer as it's text data
        # that will be encoded by CLIP and should not be normalized.
            
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        if "endpose" in self.replay_buffer:
            normalizer.params_dict["endpose"] = normalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # 新的数据结构：obs和action分离
        agent_pos_obs = sample["state"]["obs"].astype(np.float32)  # (obs_horizon, D_pos)
        action_future = sample["action"]["action"].astype(np.float32)  # (action_horizon, D_action)

        obs_data = {"agent_pos": agent_pos_obs}      # obs_horizon, D_pos
        if "point_cloud" in sample:
            point_cloud_obs = sample["point_cloud"]["obs"].astype(np.float32)  # (obs_horizon, 1024, 6)
            obs_data["point_cloud"] = point_cloud_obs

        if "endpose" in sample:
            obs_data["endpose"] = sample["endpose"]["obs"].astype(np.float32)

        if "short_state" in sample:
            short_state = sample["short_state"]["obs"].astype(np.float32)
            obs_data["short_state"] = short_state

        # Add language instruction to observation if available
        if self.use_language:
            for key in self.language_key:
                if key in sample:
                    instruction = np.random.choice(sample[key]["obs"])
                    assert isinstance(instruction, str)
                    obs_data[key] = instruction
        
        for key in self.loop_key:
            if key in sample:
                value = sample[key]["obs"]
                obs_data[key] = value
        data = {
            "obs": obs_data,
            "action": action_future,             # action_horizon, D_action
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        data_for_torch = {
            "obs": {k: v for k, v in data["obs"].items() if "instruction" not in k},
            "action": data["action"]
        }
        torch_data = dict_apply(data_for_torch, torch.from_numpy)
        for key in self.language_key:
            if key in data["obs"]:
                torch_data["obs"][key] = data["obs"][key]
        assert isinstance(torch_data["obs"]["agent_pos"], torch.Tensor)
        assert torch_data["obs"]["agent_pos"].ndim == 2, f"{torch_data['obs']['agent_pos'].shape}"  
        
        return torch_data
    
    @staticmethod
    def collate_fn(batch):
        """处理多层嵌套字典的自定义collate函数"""
        def recursive_collate(items):
            # 如果是字典类型，递归处理每个值
            if isinstance(items[0], dict):
                collated = {}
                for key in items[0].keys():
                    # 收集所有样本中该key对应的value
                    values = [item[key] for item in items]
                    collated[key] = recursive_collate(values)
                return collated
            # 如果是张量类型
            elif isinstance(items[0], torch.Tensor):
                # 检查所有张量形状是否一致
                shapes = [item.shape for item in items]
                if all(shape == shapes[0] for shape in shapes):
                    return torch.stack(items)
                else:
                    return items  # 形状不一致则返回列表
            # 其他类型使用默认collate处理
            else:
                try:
                    return default_collate(items)
                except:
                    return items

        return recursive_collate(batch)
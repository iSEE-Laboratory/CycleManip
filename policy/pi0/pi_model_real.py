#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实机器人的PI0模型 (单头部相机 + 单臂7自由度)
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_real_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image


class PI0Real:
    """
    真实机器人的PI0模型类
    
    与标准PI0的主要区别：
    - 只使用1个头部相机 (cam_high)
    - 只使用7个自由度 (右臂6关节 + 1夹爪)
    - 使用 aloha_real_policy 进行数据转换
    """

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        """
        初始化真实机器人的PI0模型
        
        Args:
            train_config_name: 训练配置名称 (例如 "pi0_base_aloha_real_lora")
            model_name: 模型名称
            checkpoint_id: checkpoint ID
            pi0_step: 推理步数
        """
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)
        ckpt_path = f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}"
        self.policy = _policy_config.create_trained_policy(
            config,
            ckpt_path,
            robotwin_repo_id=model_name)
        print(f"✓ Successfully loaded real robot model: {train_config_name}")
        print(f"ckpt_path: {ckpt_path}")
        
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        self.instruction = None

    def set_img_size(self, img_size):
        """
        设置图像大小
        
        Args:
            img_size: 图像尺寸 (height, width)
        """
        self.img_size = img_size

    def set_language(self, instruction):
        """
        设置语言指令
        
        Args:
            instruction: 任务指令字符串
        """
        self.instruction = instruction
        print(f"✓ Successfully set instruction: {instruction}")

    def update_observation_window(self, img_arr, state):
        """
        更新观察窗口缓冲区 (真实机器人只有1个相机)
        
        Args:
            img_arr: 头部相机图像 (H, W, 3) 或 (3, H, W) 或图像列表
            state: 关节状态 (7,) - [6个关节 + 1个夹爪]
        """
        assert len(state) == 7, f"Expected 7 DoF for real robot, got {len(state)}"
        
        # # 只使用头部相机 (img_arr可以是单个图像或图像数组)
        # breakpoint()
        if isinstance(img_arr, list):
            img_front = img_arr[0]
        else:
            img_front = img_arr
        
        # 确保图像是 (C, H, W) 格式
        # breakpoint()
        if img_front.shape[0] != 3:
            img_front = np.transpose(img_front, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                # cam_left_wrist 和 cam_right_wrist 会由 AlohaRealInputs 自动填充零图像
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        """
        获取动作
        
        Returns:
            actions: 动作序列，形状为 (action_horizon, 7)
        """
        assert self.observation_window is not None, "Please update observation_window first!"
        assert self.instruction is not None, "Please set language instruction first!"
        
        actions = self.policy.infer(self.observation_window)["actions"]
        
        # 动作会被 AlohaRealOutputs 自动截取为前7维
        assert actions.shape[-1] == 7, f"Expected 7 DoF actions, got {actions.shape[-1]}"
        
        return actions

    def reset_observation_windows(self):
        """重置观察窗口和语言指令"""
        self.instruction = None
        self.observation_window = None
        print("✓ Successfully reset observation windows and language instruction")


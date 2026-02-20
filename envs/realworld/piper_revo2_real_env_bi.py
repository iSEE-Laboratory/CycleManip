"""
Piper Real Environment for Policy Deployment
提供与 Base_Task 兼容的接口，用于真机部署
"""
import time
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import cv2
import open3d as o3d
from termcolor import cprint

from .piper import PiperRobot
from .camera import RealSenseCamera, get_device_ids
import pyrealsense2 as rs

from .revo2.revo2Controler import Revo2HandController

import torch
import sys
sys.path.append('/home/dex/haoran/gello_software/third_party/pointnet2')
import pointnet2_utils

"""
双手Piper 真机环境类
"""


class PiperRealEnv:
    """Piper 真机环境类
    
    提供与仿真环境 Base_Task 相似的接口，使得所有 policy 可以无缝迁移到真机
    """
    
    def __init__(
        self,
        policy: str = "unknown_policy",
        robot_ip: str = "can_right",
        init_pos = [0, 0, 0, 0, 0, 0, 0],
        step_lim: int = 1000,
        img_size: tuple = (640, 480),
    ):
        """
        Args:
            camera_config: 相机配置字典，包含各个相机的 device_id
            robot_ip: 机器人 CAN 接口名称
            step_lim: 最大执行步数
            img_size: 图像尺寸 (width, height)
        """
        cprint("=" * 50, "cyan")
        cprint(f"初始化 Piper {policy} 真机环境...", "cyan", attrs=["bold"])
        cprint("=" * 50, "cyan")

        self.policy = policy

        # 初始化手
        self.hand_left = Revo2HandController(port='/dev/ttyUSB1', slave_id=0x7e)  # 左手
        self.hand_right = Revo2HandController(port='/dev/ttyUSB0', slave_id=0x7f)  # 右手
        
        # 初始化机器人
        cprint(f"🤖 连接机器人: can_left 和 can_right", "yellow")
        self.robot_left = PiperRobot(robot_ip="can_left")
        self.robot_right = PiperRobot(robot_ip="can_right")

        self.init_pos = np.array(init_pos)
        
        print("初始化相机...")
        # d455
        # self.camera = RealSenseCamera(device_id='215122251612', flip=False)
        # self.intrinsics = rs.intrinsics()
        # self.intrinsics.width, self.intrinsics.height = 640, 480
        # self.intrinsics.ppx, self.intrinsics.ppy = 323.6994934082031, 240.37545776367188
        # self.intrinsics.fx, self.intrinsics.fy = 382.5924072265625, 382.1819763183594
        # self.intrinsics.model = rs.distortion.brown_conrady
        # self.intrinsics.coeffs = [-0.05781254917383194, 0.07238195091485977, 0.00010194736387347803,
        #                     0.0006292760954238474, -0.023512376472353935]

        self.camera = RealSenseCamera(device_id='f1271156', flip=False)
        # L515-depth
        # self.intrinsics = rs.intrinsics()
        # self.intrinsics.width, self.intrinsics.height = 640, 480
        # self.intrinsics.ppx, self.intrinsics.ppy = 301.09375, 246.337890625
        # self.intrinsics.fx, self.intrinsics.fy = 459.8203125, 459.96484375
        # self.intrinsics.model = rs.distortion.none
        # self.intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        # L515-rgb
        self.intrinsics = rs.intrinsics()
        self.intrinsics.width, self.intrinsics.height = 640, 480
        self.intrinsics.ppx, self.intrinsics.ppy = 330.53131103515625, 232.83041381835938
        self.intrinsics.fx, self.intrinsics.fy = 598.9841918945312, 599.3632202148438
        self.intrinsics.model = rs.distortion.brown_conrady
        self.intrinsics.coeffs = [0.16919225454330444, -0.5201395750045776, -0.0035975882783532143, -0.00044879087363369763, 0.4867783486843109]

        # GPU旋转
        theta_x = torch.deg2rad(torch.tensor(140., device='cuda'))
        theta_z = torch.deg2rad(torch.tensor(2.5, device='cuda'))
        R_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(theta_x), -torch.sin(theta_x)],
            [0, torch.sin(theta_x), torch.cos(theta_x)]
        ], device='cuda')
        R_z = torch.tensor([
            [torch.cos(theta_z), -torch.sin(theta_z), 0],
            [torch.sin(theta_z), torch.cos(theta_z), 0],
            [0, 0, 1]
        ], device='cuda')
        R = R_z @ R_x
        self.R = R.T

        
        self.img_size = img_size

        self.instruction = None
        self.instruction_sim = None
        self.instruction_int = None
        
        # 环境状态
        self.step_lim = step_lim
        self.take_action_cnt = 0
        self.eval_success = False
        self.suc = 0
        self.test_num = 0

        self.first_time = True
        self.reset()
        
        cprint("✅ Piper 真机环境初始化完成!", "green", attrs=["bold"])
        cprint("=" * 50, "cyan")
    
    def dofs(self) -> int:
        return 24
    
    def get_pcd(self, color_image, depth_image, intrinsics, device='cuda'):
        """GPU加速版，从RGB-D生成点云"""
        # 转tensor
        color = torch.from_numpy(color_image.copy()).float().to(device)
        depth = torch.from_numpy(depth_image.copy()).float().to(device) * 0.001

        H, W = depth.shape
        v, u = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        valid = depth > 1e-5
        z = depth[valid]
        x = (u[valid] - intrinsics.ppx) * z / intrinsics.fx
        y = (v[valid] - intrinsics.ppy) * z / intrinsics.fy

        points = torch.stack((x, y, z), dim=-1)
        colors = color[valid]

        points = points @ self.R

        valid = \
            (points[:, 0] < 2) & \
            (points[:, 1] > -3.65) & \
            (points[:, 2] > -5.75) & (points[:, 2] < -2)

        points, colors = points[valid], colors[valid]

        # print(f"点云原始点数: {points.shape[0]}")

        if points.shape[0] > 2048:
            idx1 = torch.where(points[:, 1] > -3.6)[0]
            idx2 = torch.where((points[:, 1] <= -3.6) & (points[:, 1] > -3.65))[0]

            num1 = int(2048 * 0.75)
            num2 = 2048 - num1

            p1 = points[idx1].unsqueeze(0)
            p2 = points[idx2].unsqueeze(0)
            inds1 = pointnet2_utils.furthest_point_sample(p1, num1)
            inds2 = pointnet2_utils.furthest_point_sample(p2, num2)

            sampled_points = torch.cat([
                p1[0, inds1[0]], p2[0, inds2[0]]
            ], dim=0)
            sampled_colors = torch.cat([
                colors[idx1][inds1[0]], colors[idx2][inds2[0]]
            ], dim=0)

            idx = torch.randperm(2048, device=device)
            points = sampled_points[idx]
            colors = sampled_colors[idx]

        elif points.shape[0] < 2048:
            num_pad = 2048 - points.shape[0]
            pad_points = torch.zeros((num_pad, 3), device=device)
            pad_colors = torch.zeros((num_pad, 3), device=device)
            points = torch.cat([points, pad_points], dim=0)
            colors = torch.cat([colors, pad_colors], dim=0)
            idx = torch.randperm(2048, device=device)
            points = points[idx]
            colors = colors[idx]# 用以调试，保存观测到本地
            # import pickle as pkl
            # # 保存到/home/dex/haoran/LoopBreaker/data/tmp
            # pkl.dump(obs, open(f"/home/dex/haoran/LoopBreaker/data/tmp/piper_real_dp3_obs_step{self.take_action_cnt}.pkl", "wb"))

            # input("[obs] 查看观测，按回车继续...")

            # print("state:", joint_positions)
            # print("ee_pos_quat:", ee_pos_quat)

            # input("[obs] 查看完毕，按回车继续...")

        # 只在最后转换为numpy返回
        return torch.cat([points, colors], dim=-1).cpu().numpy()

    
    def get_obs(self) -> Dict:
        """获取当前观测
        
        Returns:
            {
                'joint_action': {
                    'vector': list[24],  # (6个关节角度 + 6个手位置) x 2 手
                },
                'endpose': np.ndarray(14),  # (位置+四元数) x 2 手
                'pointcloud': np.ndarray(2048, 6)
            }
        """
        # 根据policy构建观测字典
        if self.policy == "DP3":
            # DP3 需要 joint_action.vector 和 pointcloud
            # 对于yl的模型，我们还需要传入 instruction，instruction_sim，instruction_int, ee_pos_quat
            # breakpoint()
            robot_obs_left = self.robot_left.get_observations()
            robot_obs_right = self.robot_right.get_observations()
            # 原来拿到的是7，这里没有夹爪，所以只取前6个
            joint_positions_left = robot_obs_left["joint_positions"][:6]  # shape: (6,)
            ee_pos_quat_left = robot_obs_left["ee_pos_quat"]  # shape: (7,)
            joint_positions_right = robot_obs_right["joint_positions"][:6]  # shape: (6,)
            ee_pos_quat_right = robot_obs_right["ee_pos_quat"]  # shape: (7,)

            hand_state_left = np.array(self.hand_left.get_joint_positions(), dtype=np.float32) # (6,)
            # debug
            # hand_state_left = np.array([0, 1, 0, 0, 0, 0], dtype=np.float32)
            hand_state_right = np.array(self.hand_right.get_joint_positions(), dtype=np.float32) # (6,)
            # hand_state_right = np.array(self.init_pos[18:24], dtype=np.float32)

            joint_positions = np.concatenate([joint_positions_left, hand_state_left, joint_positions_right, hand_state_right])
            assert joint_positions.shape == (24,)
            ee_pos_quat = np.concatenate([ee_pos_quat_left, ee_pos_quat_right])
            assert ee_pos_quat.shape == (14,)

            # 获取相机图像
            rgb, depth = self.camera.read(img_size=self.img_size)
            depth = depth.reshape(480, 640)
            pcd = self.get_pcd(rgb, depth, self.intrinsics)  # shape: (2048, 6)

            obs = {
                "joint_action": {
                    "vector": np.array(joint_positions),  # 返回 numpy array，不是 list！
                },
                "endpose": np.array(ee_pos_quat).astype(np.float32),  # 返回 numpy array，不是 list！

                "pointcloud": pcd,

                # instructions
                "instruction": self.instruction,
                "instruction_sim": self.instruction_sim,
                "instruction_int": self.instruction_int
            }

            # # 用以调试，保存观测到本地
            # import pickle as pkl
            # # 保存到/home/dex/haoran/LoopBreaker/data/tmp
            # pkl.dump(obs, open(f"/home/dex/haoran/LoopBreaker/data/tmp/piper_real_dp3_obs_step{self.take_action_cnt}.pkl", "wb"))

            # input("[obs] 查看观测，按回车继续...")

            # print("state:", joint_positions)
            # print("ee_pos_quat:", ee_pos_quat)

            # input("[obs] 查看完毕，按回车继续...")
            


        elif self.policy == "pi0":
            # pi0 需要 joint_action.vector 和 head_camera.rgb
            robot_obs_left = self.robot_left.get_observations()
            robot_obs_right = self.robot_right.get_observations()
            joint_positions_left = robot_obs_left["joint_positions"]  # shape: (7,)
            joint_positions_right = robot_obs_right["joint_positions"]  # shape: (7,)

            joint_positions = np.concatenate([joint_positions_left, joint_positions_right])

            # 获取相机图像
            rgb, _ = self.camera.read(img_size=self.img_size)
            
            obs = {
                "joint_action": {
                    "vector": np.array(joint_positions),  # 返回 numpy array，不是 list！
                },
                "observation": {
                    "head_camera": {
                        "rgb": rgb,
                    },
                },  
            }

        else:
            raise ValueError(f"未支持的 policy 类型: {self.policy}")
            
        return obs
    
    def take_action(self, action: np.ndarray) -> None:
        """执行动作
        
        Args:
            action: 动作数组，shape: (24,)
                - 左臂6个关节弧度 + 左手6个关节(0-1000整数) + 右臂6个关节弧度 + 右手6个关节(0-1000整数)
        """
        self.take_action_cnt += 1
             
        # 确保是 numpy 数组
        action = np.array(action, dtype=np.float32)

        joint_left = action[:6]
        joint_left[5] += 0.4101523743

        joint_right = action[12:18]
        joint_right[5] += 0.4101523743

        left_arm_cmd = np.concatenate([joint_left, np.array([0])])
        left_hand_cmd = action[6:12]  # 手6个关节 (原始值 0-1000)
        right_arm_cmd = np.concatenate([joint_right, np.array([0])])
        right_hand_cmd = action[18:24]  # 手6个关节 (原始值 0-1000)

        # 发送指令到机器人
        self.robot_left.command_joint_state(left_arm_cmd, "state")
        self.robot_right.command_joint_state(right_arm_cmd, "state")
        self.hand_left.set_joint_positions(left_hand_cmd.astype(int).tolist())
        self.hand_right.set_joint_positions(right_hand_cmd.astype(int).tolist())
        
        cprint(f"⏳ 步数: {self.take_action_cnt}/{self.step_lim}", "cyan", end="\r")
      
    def reset(self) -> None:
        """重置机器人到初始姿态"""
        # 用一个线性插值去控制夹爪到初始位置，而不是直接跳到初始位置
        joint_left = self.init_pos[0:6]
        joint_left[5] += 0.4101523743
        left_arm_init = np.concatenate([joint_left, np.array([0])])
        # print(left_arm_init.shape)

        joint_right = self.init_pos[12:18]
        # joint_right[5] += 0.4101523743
        right_arm_init = np.concatenate([joint_right, np.array([0])])

        for i in range(100):
            alpha = (i + 1) / 100.0
            interp_pos_left = (1 - alpha) * self.robot_left.get_joint_state() + alpha * left_arm_init
            self.robot_left.command_joint_state(interp_pos_left, "state")
            interp_pos_right = (1 - alpha) * self.robot_right.get_joint_state() + alpha * right_arm_init
            self.robot_right.command_joint_state(interp_pos_right, "state")
            time.sleep(0.03)
        # self.robot.command_joint_state(self.init_pos)

        self.hand_left.set_joint_positions(self.init_pos[6:12].astype(int).tolist())
        self.hand_right.set_joint_positions(self.init_pos[18:24].astype(int).tolist())

        self.take_action_cnt = 0
        self.eval_success = False
        time.sleep(0.5)
        cprint("✅ 已将piper重置到初始位置", "green")

    def set_instruction(self, instruction: str, instruction_int: str = None, instruction_sim: str = None) -> None:
        """设置任务指令（语言描述）"""
        self.instruction = instruction
        self.instruction_int = instruction_int
        self.instruction_sim = instruction_sim
        if instruction is not None:
            cprint(f"📝 任务指令: {instruction}", "blue")
        if instruction_int is not None:
            cprint(f"📝 任务指令_int: {instruction_int}", "blue")
        if instruction_sim is not None:
            cprint(f"📝 任务指令_sim: {instruction_sim}", "blue")
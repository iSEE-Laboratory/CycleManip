"""
Piper Real Environment for Policy Deployment
提供与 Base_Task 兼容的接口，用于真机部署
"""
import time
import numpy as np
from typing import Any, Dict, Optional, List
from pathlib import Path
import cv2
import open3d as o3d
from termcolor import cprint

from .piper import PiperRobot
from .camera import RealSenseCamera, get_device_ids
import pyrealsense2 as rs


import torch
import sys
sys.path.append('/home/dex/haoran/gello_software/third_party/pointnet2')
import pointnet2_utils


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
        
        # 初始化机器人
        cprint(f"🤖 连接机器人: {robot_ip}", "yellow")
        self.robot = PiperRobot(robot_ip=robot_ip)

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
        return 7
    
    def _load_data_to_memory(self):
        """将HDF5数据预加载到内存中，并解码RGB图像"""

        self.data_path = "/home/dex/haoran/gello_software/data_processed_10hz/test_bbhlr_enhance/hands_10hz/data/episode0.hdf5"


        import h5py
        with h5py.File(self.data_path, 'r') as h5_file:
            # 获取数据长度
            self.data_length = len(h5_file['joint_action']['vector'])
            
            # 预加载关节状态数据
            self.joint_state_array = np.array(h5_file['joint_state']['vector'][:])

            self.endpose_array = np.array(h5_file['endpose'][:])
            
            # 预加载并解码RGB图像数据
            # 预加载点云数据
            self.pointcloud_array = np.array(h5_file['pointcloud'][:])
            
            # 预加载动作数据（用于对比）
            self.joint_action_array = np.array(h5_file['joint_state']['vector'][:])
            
        cprint("\n数据结构:", "yellow")
        cprint(f"  - joint_state_array: {self.joint_state_array.shape}", "white")
        cprint(f"  - pointcloud_array: {self.pointcloud_array.shape}", "white")
        cprint(f"  - joint_action_array: {self.joint_action_array.shape}", "white")
        print()

    def get_obs_dataset(self) -> Dict[str, Any]:
        """
        获取当前步的观测数据（从内存中读取）
        
        Returns:
            observation: 包含 point_cloud 和 agent_pos 的字典
        """
        step = min(self.take_action_cnt, self.data_length - 1)
        
        cprint(f"📸 从数据集中获取观测: step {step}/{self.data_length}", "cyan")
        

        # DP3 需要 joint_action.vector 和 pointcloud
        obs = {
            "joint_action": {},
            "endpose": None,
            "pointcloud": None,
            "instruction": None,
            "instruction_sim": None,
            "instruction_int": None
        }
        
        # 1. 读取关节状态
        joint_vector = self.joint_state_array[step]
        obs["joint_action"]["vector"] = np.array(joint_vector)

        # 2. 读取点云
        pointcloud = self.pointcloud_array[step]
        obs["pointcloud"] = np.array(pointcloud)

        endpose = self.endpose_array[step]
        obs["endpose"] = np.array(endpose)

        # 3. 读取指令（如果有）
        if self.instruction is not None:
            obs["instruction"] = self.instruction
        if self.instruction_sim is not None:
            obs["instruction_sim"] = self.instruction_sim
        if self.instruction_int is not None:
            obs["instruction_int"] = self.instruction_int
            
        # 缓存观测
        self._current_obs = obs
        
        return obs
    
    def take_action_dataset(self):
        """
        获取当前步的动作数据（从内存中读取）
        
        Returns:
            action: 当前步的动作数组
        """
        step = min(self.take_action_cnt, self.data_length - 1)
        
        # if self.verbose:
        cprint(f"🤖 获取动作: step {step}/{self.data_length}", "cyan")
        
        action = self.joint_action_array[step]
        
        # return action
        self.take_action_cnt += 1
        cprint(f"⏳ 步数: {self.take_action_cnt}/{self.step_lim}", "cyan", end="\r")
        self.robot.command_joint_state(action, "state")


    
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

        # 筛选（同样在GPU上）
        valid = \
            (points[:, 0] < 2) & \
            (points[:, 1] > -3.65) & \
            (points[:, 2] > -5.75) & (points[:, 2] < -2) & \
            ~((points[:, 0] > 0.8) & (points[:, 1] > -3.57) & (points[:, 2] < -2.5))

        # valid = \
        #     (points[:, 0] < 2) & \
        #     (points[:, 1] > -3.65) & \
        #     (points[:, 2] > -5.75) & (points[:, 2] < -2)

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
            colors = colors[idx]

        # 只在最后转换为numpy返回
        return torch.cat([points, colors], dim=-1).cpu().numpy()

    
    def get_obs(self) -> Dict:
        """获取当前观测
        
        Returns:
            与仿真环境格式一致的观测字典:
            {
                'joint_action': {
                    'right_arm': list[6],  # Piper 单臂，填充空值
                    'right_gripper': float,
                    'vector': list[7],  # 6 joints + 1 gripper
                },
                'pointcloud': np.ndarray(2048, 6)
            }
        """
        # 根据policy构建观测字典
        if self.policy == "DP3":
            # DP3 需要 joint_action.vector 和 pointcloud
            # 对于yl的模型，我们还需要传入 instruction，instruction_sim，instruction_int, ee_pos_quat
            robot_obs = self.robot.get_observations()
            joint_positions = robot_obs["joint_positions"]  # shape: (7,)
            ee_pos_quat = robot_obs["ee_pos_quat"]  # shape: (7,)

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
            # 保存到/home/dex/haoran/LoopBreaker/data/tmp
            # import pickle as pkl
            # pkl.dump(obs, open(f"/home/dex/haoran/LoopBreaker/data/tmp/piper_real_dp3_obs_step{self.take_action_cnt}.pkl", "wb"))


        elif self.policy == "pi0":
            # pi0 需要 joint_action.vector 和 head_camera.rgb
            robot_obs = self.robot.get_observations()
            joint_positions = robot_obs["joint_positions"]  # shape: (7,)

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
            action: 目标关节状态，shape: (7,) 或 (14,)
                   - 如果是 (7,): [6个关节角度 + 1个夹爪位置]
                   - 如果是 (14,): [左臂6+左夹爪1 + 右臂6+右夹爪1]，只使用前7个
        """
        self.take_action_cnt += 1
             
        # 确保是 numpy 数组
        action = np.array(action)

        print(f"🤖 执行动作: {action}")
        
        # 发送指令到机器人
        self.robot.command_joint_state(action, "state")
        
        cprint(f"⏳ 步数: {self.take_action_cnt}/{self.step_lim}", "cyan", end="\r")
      
    def reset(self) -> None:
        """重置机器人到初始姿态"""
        # 用一个线性插值去控制夹爪到初始位置，而不是直接跳到初始位置
        is_replace = input("是否放锤子？(y/n)")
        if is_replace.lower() == 'y':
            cprint("放锤子中...", "yellow")
            target = [ -2905, 111154, -59434,  -3563,  -1365,  24480, 0]
            # target = np.array()[0.12275021 -0.01490509 -0.23432243 -0.12770648  0.385226    0.44116578]
            # target = [19077, 113634, -38837, -12259, -53672, 20033, 0]
            for i in range(50):
                alpha = (i + 1) / 50.0
                interp_pos = (1 - alpha) * self.robot.get_joint_state() + alpha * np.array(target)
                # self.robot.command_joint_state(interp_pos, "state")
                joint_state_int = (interp_pos[:6]).astype(int)
                target_piper_angle = interp_pos[6]
                self.robot.piper.JointCtrl(*joint_state_int)
                self.robot.piper.GripperCtrl(
                    gripper_angle=int(target_piper_angle), 
                    gripper_effort=1000, 
                    gripper_code=0x01
                )
                time.sleep(0.03)

                # 夹爪回到初始位置

            time.sleep(0.5)

            target = [ -2905, 111154, -59434,  -3563,  -1365,  24480, 70000]
            # target = [19077, 113634, -38837, -12259, -53672, 20033, 70000]

            for i in range(30):
                alpha = (i + 1) / 30.0
                interp_pos = (1 - alpha) * self.robot.get_joint_state() + alpha * np.array(target)
                joint_state_int = (interp_pos[:6]).astype(int)
                target_piper_angle = interp_pos[6]
                # self.robot.piper.JointCtrl(*joint_state_int)
                self.robot.piper.GripperCtrl(
                    gripper_angle=int(target_piper_angle), 
                    gripper_effort=1000, 
                    gripper_code=0x01
                )
                time.sleep(0.03)

            time.sleep(0.8)

        for i in range(100):
            alpha = (i + 1) / 100.0
            interp_pos = (1 - alpha) * self.robot.get_joint_state() + alpha * self.init_pos
            self.robot.command_joint_state(interp_pos, "state")
            time.sleep(0.03)
        # self.robot.command_joint_state(self.init_pos)
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
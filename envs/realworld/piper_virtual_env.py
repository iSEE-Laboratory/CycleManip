"""
Piper Virtual Environment for Testing Real World Pipeline
虚拟的真机环境,用于测试真机部署 pipeline
从 HDF5 数据文件中读取观测，模拟真机执行流程
"""
import h5py
import numpy as np
import cv2
from typing import Optional, Dict, Any
from pathlib import Path
import time
from termcolor import cprint


class PiperVirtualEnv:
    """
    虚拟的 Piper 真机环境，用于测试真机部署 pipeline
    从 HDF5 数据中读取观测数据，不实际连接机器人
    """
    
    def __init__(
        self,
        policy: str = "unknown_policy", 
        data_path: str = "",
        step_lim: int = 500,
        verbose: bool = True
    ):
        """
        初始化虚拟环境
        
        Args:
            data_path: HDF5 数据文件路径
            step_lim: 最大步数限制
            verbose: 是否打印详细信息
        """
        self.policy = policy
        self.data_path = Path(data_path)
        self.step_lim = step_lim
        self.verbose = verbose
        
        # 预加载所有数据到内存
        cprint(f"📂 加载数据文件: {self.data_path}", "cyan")
        self._load_data_to_memory()
        
        cprint(f"✅ 数据加载完成，共 {self.data_length} 帧", "green")
        
        # 状态变量
        self.current_step = 0
        self.take_action_cnt = 0
        self.test_num = 0
        self.suc = 0
        self.eval_success = False
        self.instruction = None
        
        # 缓存当前观测
        self._current_obs = None
        self.result = []
        
    def _load_data_to_memory(self):
        """将HDF5数据预加载到内存中，并解码RGB图像"""
        with h5py.File(self.data_path, 'r') as h5_file:
            # 获取数据长度
            self.data_length = len(h5_file['joint_action']['vector'])
            
            # 预加载关节状态数据
            self.joint_state_array = np.array(h5_file['joint_state']['vector'][:])
            
            # 预加载并解码RGB图像数据
            self.head_rgb_array = self._decode_rgb_images(h5_file['observation']['head_camera']['rgb'])
            
            # 预加载点云数据
            self.pointcloud_array = np.array(h5_file['pointcloud'][:])
            
            # 预加载动作数据（用于对比）
            self.joint_action_array = np.array(h5_file['joint_action']['vector'][:])
            
        cprint("\n数据结构:", "yellow")
        cprint(f"  - joint_state_array: {self.joint_state_array.shape}", "white")
        cprint(f"  - head_rgb_array: {self.head_rgb_array.shape}", "white")
        cprint(f"  - pointcloud_array: {self.pointcloud_array.shape}", "white")
        cprint(f"  - joint_action_array: {self.joint_action_array.shape}", "white")
        print()
    
    def _decode_rgb_images(self, rgb_dataset):
        """解码RGB图像数据集"""
        decoded_images = []
        
        for i in range(len(rgb_dataset)):
            # 获取二进制数据
            camera_bits = rgb_dataset[i]
            
            # 解码JPEG图像
            camera_img = cv2.imdecode(np.frombuffer(camera_bits, np.uint8), cv2.IMREAD_COLOR)
            
            # 转换BGR到RGB
            camera_rgb = cv2.cvtColor(camera_img, cv2.COLOR_BGR2RGB)
            decoded_images.append(camera_rgb)
        
        return np.array(decoded_images)
    
    def get_obs(self) -> Dict[str, Any]:
        """
        获取当前步的观测数据（从内存中读取）
        
        Returns:
            observation: 包含 point_cloud 和 agent_pos 的字典
        """
        step = min(self.current_step, self.data_length - 1)
        
        if self.verbose:
            cprint(f"📸 获取观测: step {step}/{self.data_length}", "cyan")
        
        # 根据policy构建观测字典
        if self.policy == "pi0":
            # pi0 需要 joint_action.vector 和 head_camera.rgb
            obs = {
                "joint_action": {},
                "observation": {
                    "head_camera": {
                        "rgb": None,
                    },
                },
            }
            
            # 1. 读取关节状态
            joint_vector = self.joint_state_array[step]
            obs["joint_action"]["vector"] = np.array(joint_vector)

            # 2. 读取RGB图像（已解码）
            obs["observation"]["head_camera"]["rgb"] = self.head_rgb_array[step]

        elif self.policy == "DP3":
            # DP3 需要 joint_action.vector 和 pointcloud
            obs = {
                "joint_action": {},
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

            # 3. 读取指令（如果有）
            if self.instruction is not None:
                obs["instruction"] = self.instruction
            if self.instruction_sim is not None:
                obs["instruction_sim"] = self.instruction_sim
            if self.instruction_int is not None:
                obs["instruction_int"] = self.instruction_int

        else:
            raise ValueError(f"未支持的 policy 类型: {self.policy}")
        
        # 缓存观测
        self._current_obs = obs
        
        return obs
    
    def take_action(self, action: np.ndarray) -> None:
        """
        执行动作（虚拟执行，不实际控制机器人）
        
        Args:
            action: 动作向量 [joint1, ..., joint7]
        """
        self.take_action_cnt += 1
        
        cprint(f"🎮 Step {self.take_action_cnt}/{self.step_lim}: 执行动作 {action}", "magenta")

        self.result.append(action.tolist())
        
        # 更新到下一步
        self.current_step += 1
    
    def reset(self) -> None:
        """重置环境到初始状态"""
        if self.verbose:
            cprint("\n🔄 重置环境", "yellow")
        
        self.current_step = 0
        self.take_action_cnt = 0
        self.eval_success = False
        self._current_obs = None
        
        if self.verbose:
            cprint("✅ 环境已重置到初始状态\n", "green")

    def set_instruction(self, instruction: str, instruction_int: str = None, instruction_sim: int = None) -> None:
        """设置任务指令"""
        self.instruction = instruction
        self.instruction_int = instruction_int
        self.instruction_sim = instruction_sim
        if self.verbose:
            if instruction:
                cprint(f"📝 设置任务指令: {instruction}", "blue")
            if instruction_int:
                cprint(f"📝 设置任务指令_int: {instruction_int}", "blue")
            if instruction_sim:
                cprint(f"📝 设置任务指令_sim: {instruction_sim}", "blue")
    
    def get_instruction(self) -> str:
        """获取任务指令"""
        return self.instruction if self.instruction else "Complete the task"
    
    def _manual_check_success(self) -> bool:
        """手动检查任务是否成功（用于测试）"""
        cprint("\n" + "=" * 60, "yellow")
        cprint("任务是否成功完成?", "yellow")
        response = input("请输入 (y/n): ").strip().lower()
        success = response == 'y'
        return success
    
    def close(self) -> None:
        """关闭环境并释放资源"""
        # 对比模型推理出来的action与真实action的差异
        # if len(self.result) > 0:
        #     result_array = np.array(self.result)
        #     true_actions = self.joint_action_array[:len(result_array)]
            
        #     # 每一条打印出来
        #     cprint("\n" + "=" * 60, "yellow", attrs=["bold"])
        #     cprint("📊 动作对比分析", "yellow", attrs=["bold"])
        #     cprint("=" * 60, "yellow", attrs=["bold"])

        #     for i in range(len(self.result)):  # 全部显示
        #         cprint(f"Step {i+1}:", "white", attrs=["bold"])
        #         if i < len(true_actions):
        #             cprint(f"  Model: {self.result[i]}", "cyan")
        #             cprint(f"  True:  {true_actions[i].tolist()}", "yellow")
        #         else:
        #             cprint(f"  Model: {self.result[i]}", "cyan")


        # # 把结果保存到同目录下的 result.npy 文件中
        # if len(self.result) > 0:
        #     result_path = self.data_path.parent / (self.data_path.stem + "_result.npy")
        #     np.save(result_path, np.array(self.result))
        #     cprint(f"💾 结果已保存到: {result_path}", "green")
        pass
    
    def __del__(self):
        """析构函数"""
        self.close()
    
    def __repr__(self) -> str:
        return (f"PiperVirtualEnv(data={self.data_path.name}, "
                f"length={self.data_length}, step={self.current_step})")

"""
Piper Real Environment for Policy Deployment
提供与 Base_Task 兼容的接口，用于真机部署
"""
import time
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import cv2
from termcolor import cprint

import socket
import pickle

def recv_all(sock, n):
    """辅助函数：接收n字节的完整数据"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


INIT_POS = np.array([0.0, -0.0, -0.0, 0.7020077109336853] + 
                    [-0.341080171311016, 0.33767049909606506, 0.006908714791052963, 0.09343911764866444, 
                     0.7626252197300604, -0.8777070566538752, 0.04993957316833356, -0.2814281473516445, 
                     -0.16155323341763803, 0.4014385403063156, -0.3524181486645954, -1.1506913605633977, 
                     -0.07861215198585017, 0.25988312787405904]) 

class HumanoidRealEnv:
    """Piper 真机环境类
    
    提供与仿真环境 Base_Task 相似的接口，使得所有 policy 可以无缝迁移到真机
    """
    
    def __init__(
        self,
        policy: str = "unknown_policy",
        robot_ip: str = "can_right",
        init_pos = INIT_POS,
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

        
        # SERVER_IP = '192.168.31.69'  # 改成 server 的 IP
        SERVER_IP = '192.168.123.164'  # 改成 server 的 IP
        PORT = 5000

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((SERVER_IP, PORT))
        print(f"[*] Connected to server {SERVER_IP}:{PORT}")
        
        # 初始化机器人
        self.init_pos = np.array(init_pos)
        self.latet_robot_state = self.init_pos.copy()

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
        
        cprint("✅ humanoid 真机环境初始化完成!", "green", attrs=["bold"])
        cprint("=" * 50, "cyan")
    
    def dofs(self) -> int:
        return 7
    
    def send_msg_and_receive(self, msg: Dict) -> Dict:
        # 序列化消息
        msg_data = pickle.dumps(msg)
        msg_len = len(msg_data)
        # 发送消息长度和消息内容
        self.s.sendall(msg_len.to_bytes(4, 'big') + msg_data)
        # 接收回复长度
        raw_replylen = recv_all(self.s, 4)
        if not raw_replylen:
            raise ConnectionError("未收到服务器回复")
        replylen = int.from_bytes(raw_replylen, 'big')
        # 接收完整回复
        reply_data = recv_all(self.s, replylen)
        if not reply_data:
            raise ConnectionError("未收到服务器回复")
        # 反序列化回复
        reply = pickle.loads(reply_data)
        return reply
  
    def get_obs(self) -> Dict:
        msg = {'type': 'obs'}

        reply = self.send_msg_and_receive(msg)
        
        if reply is not None:
            # print(reply)
            if 'data' in reply:
                robot_state = reply['data']['joints_obs'] # (27,)
                # robot_state = np.array([0.0] * 27)
                self.latet_robot_state = robot_state if robot_state is not None else self.latet_robot_state
            else:
                cprint("未收到机器人状态数据，保持上一次状态", "red")
                

        # print("机器人状态:", robot_state)
        # print("机器人状态形状:", np.array(robot_state).shape)

        # 获取相机图像
        obs = {
            "joint_action": {
                "vector": np.array(self.latet_robot_state),  # 返回 numpy array，不是 list！
            },

            "pointcloud": np.zeros((3, 3)),  # 占位，后续添加真实点云数据
            "endpose": np.zeros((7,)),  # 占位，后续添加真实末端位姿数据

            # instructions
            "instruction": self.instruction,
            "instruction_sim": self.instruction_sim,
            "instruction_int": self.instruction_int
        }

        # print(obs)
        # 保存到/home/dex/haoran/LoopBreaker/data/tmp
        # import pickle as pkl
        # pkl.dump(obs, open(f"/home/dex/haoran/LoopBreaker/data/tmp/piper_real_dp3_obs_step{self.take_action_cnt}.pkl", "wb"))
        return obs
    
    def take_action(self, action: np.ndarray) -> None:
        """执行动作
        
        Args:
            action: 目标关节状态，shape: (7,) 或 (14,)
                   - 如果是 (7,): [6个关节角度 + 1个夹爪位置]
                   - 如果是 (14,): [左臂6+左夹爪1 + 右臂6+右夹爪1]，只使用前7个
        """
        self.take_action_cnt += 1
             
        # print(action)
        # 确保是 numpy 数组
        msg = {'type': 'control', 'data': action.tolist()}
        
        _ = self.send_msg_and_receive(msg)
        
        cprint(f"⏳ 步数: {self.take_action_cnt}/{self.step_lim}", "cyan", end="\r")
      
    def reset(self) -> None:
        """重置机器人到初始姿态"""
        cprint("\n🔄 重置机器人到初始姿态...", "yellow", attrs=["bold"])
        
        msg = {'type': 'control', 'data': self.init_pos.tolist()}
        
        _ = self.send_msg_and_receive(msg)
        
        self.take_action_cnt = 0
        time.sleep(1.0)  # 等待一段时间让机器人稳定
        
        cprint("✅ 机器人重置完成!", "green", attrs=["bold"])
        pass
        

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


if __name__ == "__main__":
    env = HumanoidRealEnv(policy="test_policy")
    
    # while True:
    #     env.get_obs()

    #     time.sleep(0.1)


    env.take_action(np.array([0.0, -0.0, -0.0, 0.7020077109336853] + [-0.341080171311016, 0.33767049909606506, 0.006908714791052963, 0.09343911764866444, 0.7626252197300604, -0.8777070566538752, 0.04993957316833356, -0.2814281473516445, -0.16155323341763803, 0.4014385403063156, -0.3524181486645954, -1.1506913605633977, -0.07861215198585017, 0.25988312787405904]))    


    print("Done")
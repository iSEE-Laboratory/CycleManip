import time
from typing import Dict, Literal
import numpy as np
from termcolor import cprint
from transforms3d.euler import quat2euler, euler2quat

from piper_sdk import *



class PiperRobot():
    """A class representing a Piper robot, updated for the V2 SDK based on INTERFACE_V2.MD."""

    def __init__(self, robot_ip: str = "can_right"):
        """
        Initializes the Piper robot.
        """
        self.PIPER_GRIPPER_MAX_OPEN_ANGLE = 70000.0 # API_V2_UPDATE: 使用角度作为最大开合度，假设70度为最大
        self.FACTOR = 57295.7795 # 1000*180/3.1415926, 用于将弧度转换为 Piper SDK 所需的单位 (0.001度)
        self.INTERPOLATION_STEPS = 5  # 插值步数，用于平滑运动
        
        self.piper = C_PiperInterface_V2(can_name=robot_ip)
        self.piper.ConnectPort()

        while(not self.piper.EnablePiper()):
            time.sleep(0.01)

        # mit control mode
        # self.piper.MotionCtrl_2(1, 1, 0, 0xAD)# 0xFC
        # time.sleep(1)

    def dofs(self) -> int:
        return 7
    
    def go_zero(self):
        self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)

    def get_gripper_angle(self) -> float:
        """
        获取当前夹爪的原始角度值 (0 -> 70000)。
        """
        gripper_feedback = self.piper.GetArmGripperMsgs()
        return float(gripper_feedback.gripper_state.grippers_angle)
    
    def get_joint_state(self) -> np.ndarray:
        """
        Get the current state of the Piper robot's joints.
        """
        joint_feedback_raw = self.piper.GetArmJointMsgs()
        
        # 假设 joint_feedback_raw 是一个包含6个整数的列表或元组
        # 需要将单位从 0.001度 转换为 弧度
        joint_state_data = joint_feedback_raw.joint_state
        joint_angles_raw = [
                joint_state_data.joint_1,
                joint_state_data.joint_2,
                joint_state_data.joint_3,
                joint_state_data.joint_4,
                joint_state_data.joint_5,
                joint_state_data.joint_6,
            ]
        robot_joints_rad = np.array(joint_angles_raw) * 0.001 * (np.pi / 180.0)

        current_gripper_angle = self.get_gripper_angle()
        # 归一化处理
        gripper_pos_normalized = 1.0 - (current_gripper_angle / self.PIPER_GRIPPER_MAX_OPEN_ANGLE)
        
        pos = np.append(robot_joints_rad, gripper_pos_normalized)
        return pos

    def command_joint_state(self, joint_state: np.ndarray, mode: Literal["state", "control"], interpolation_step: int = None) -> None:
        """
        向机器人发送指令。
        Args:
            joint_state (np.ndarray): 一个7维向量，前6个是手臂的目标关节角度（弧度），
                                      最后一个是夹爪的归一化目标位置（0=开, 1=闭）。
        """
        print("[arm] ", joint_state)
        # --- 控制手臂 ---
        arm_joints_rad = joint_state[:6]
        joint_state_int = (arm_joints_rad * self.FACTOR).astype(int)

        if mode == "state":
            gripper_normalized_value = 1 - joint_state[6]

            # cprint(gripper_normalized_value, "yellow")

            if gripper_normalized_value < 0.6:
                gripper_normalized_value = gripper_normalized_value * 0.1
            elif gripper_normalized_value > 0.6:
                gripper_normalized_value = min(1.5 * gripper_normalized_value, 1.0)

            # cprint(gripper_normalized_value, "blue")
        elif mode == "control":
            # 似乎是厂家工件没对齐， 关节6需要c加30000
            joint_state_int[5] += 23500
            # --- 控制夹爪 ---
            # 1. 从 Gello 获取归一化的夹爪指令 (0=开, 1=闭)
            gripper_normalized_value = joint_state[6]

        target_piper_angle = self.PIPER_GRIPPER_MAX_OPEN_ANGLE * gripper_normalized_value

        # print(joint_state_int, int(target_piper_angle))

        self.piper.JointCtrl(*joint_state_int)
        self.piper.GripperCtrl(
            gripper_angle=int(target_piper_angle), 
            gripper_effort=1000, 
            gripper_code=0x01
        )
        print("[command] ", joint_state_int, int(target_piper_angle))

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints_rad = self.get_joint_state()
   
        end_effector_raw = self.piper.GetArmEndPoseMsgs()
        pose_raw = [
            end_effector_raw.end_pose.X_axis,
            end_effector_raw.end_pose.Y_axis,
            end_effector_raw.end_pose.Z_axis,
            end_effector_raw.end_pose.RX_axis,
            end_effector_raw.end_pose.RY_axis,
            end_effector_raw.end_pose.RZ_axis,
        ]
        # 假设 end_effector_raw 是一个包含 [x, y, z, rx, ry, rz] 的列表或元组
        end_effector_pose = np.array(pose_raw) * 0.001 # 转换为 mm 和 度
        end_effector_pose[:3] /= 1000.0 # 将毫米(mm)转换为米(m)
        end_effector_pose[3:] *= (np.pi / 180.0) # 将度(°)转换为弧度(rad)
        gripper_pos_normalized = np.array([joints_rad[-1]])
        
        end_effector_pose_quat = np.zeros(7)
        end_effector_pose_quat[:3] = end_effector_pose[:3]
        end_effector_pose_quat[3:] = euler2quat(end_effector_pose[3], end_effector_pose[4], end_effector_pose[5])

        # print("[state] ", joints_rad, gripper_pos_normalized)

        return {
            "joint_positions": joints_rad,
            "joint_velocities": np.zeros_like(joints_rad), 
            "ee_pos_quat": end_effector_pose_quat,
            "ee_pose": end_effector_pose, 
            "gripper_position": gripper_pos_normalized,
        }

    def get_joint_currents(self) -> np.ndarray:
        current1 = self.piper.GetArmHighSpdInfoMsgs().motor_1.current
        current2 = self.piper.GetArmHighSpdInfoMsgs().motor_2.current
        current3 = self.piper.GetArmHighSpdInfoMsgs().motor_3.current
        current4 = self.piper.GetArmHighSpdInfoMsgs().motor_4.current
        current5 = self.piper.GetArmHighSpdInfoMsgs().motor_5.current
        current6 = self.piper.GetArmHighSpdInfoMsgs().motor_6.current
        raw_currents = [current1, current2, current3, current4, current5, current6]
        return np.array(raw_currents)

    def get_gripper_torque(self) -> int:
        return self.piper.GetArmGripperMsgs().gripper_state.grippers_effort
    

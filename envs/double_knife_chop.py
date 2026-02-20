from ._base_task import Base_Task
from .utils import *
import sapien
import math
from transforms3d.euler import quat2euler, euler2quat
import random
import numpy as np
import os


class double_knife_chop(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.board_pos = sapien.Pose([-0.02, 0.1, 0.75], [0.49574742, 0.49579477, 0.50413066, 0.5042563])
        self.board = create_actor(
            scene=self,
            pose=self.board_pos,
            modelname="131_chopping_board",
            convex=True,
            model_id=0,
            is_static=False,
        )
        self.board.set_mass(1)

        # self.load_knife([0.2, 0.15, 0.77301395], 0)
        # self.load_knife([-0.2, 0.15, 0.77301395], 1)

        # 确保刀具分配到不同的边
        # 右边的刀 - knife_1 固定在右半边
        knife_1_x = random.uniform(0.20, 0.35)  # 确保在右半边
        knife_1_y = random.uniform(0.10, 0.20)  # 随机y坐标
        knife_pose_p = [knife_1_x, knife_1_y, 0.77301395]
        knife_pose_q = [0, 0, 1, 0]

        eps = 0.03
        box1_pos_p = [knife_pose_p[0] - eps, knife_pose_p[1], knife_pose_p[2]]
        box1_pos_q = [0, 0, 0, 1]
        box2_pos_p = [knife_pose_p[0] + eps, knife_pose_p[1], knife_pose_p[2]]
        box2_pos_q = [0, 0, 0, 1]

        self.box1_1 = create_box(
            scene=self,
            pose=sapien.Pose(box1_pos_p, box1_pos_q),
            half_size=(0.01, 0.05, 0.05),
            color=(0, 0, 0),
            name="box1",
            is_static=True,
        )
        self.box1_2 = create_box(
            scene=self,
            pose=sapien.Pose(box2_pos_p, box2_pos_q),
            half_size=(0.01, 0.05, 0.05),
            color=(0, 0, 0),
            name="box2",
            is_static=True,
        )

        self.knife_1 = create_actor(
            scene=self,
            pose=sapien.Pose(knife_pose_p, knife_pose_q),
            # pose=sapien.Pose([0, -0.06, 0.6875], [0, 0, 0.995, 0.105]),
            modelname="034_knife",
            convex=True,
            model_id=0,
            is_static=False,
        )
        self.knife_1.set_mass(0.0045) 

        # 左边的刀 - knife_2 固定在左半边
        knife_2_x = random.uniform(-0.35, -0.20)  # 确保在左半边
        knife_2_y = random.uniform(0.10, 0.20)    # 随机y坐标
        knife_pose_p = [knife_2_x, knife_2_y, 0.77301395]
        knife_pose_q = [0, 0, 1, 0]

        eps = 0.03
        box1_pos_p = [knife_pose_p[0] - eps, knife_pose_p[1], knife_pose_p[2]]
        box1_pos_q = [0, 0, 0, 1]
        box2_pos_p = [knife_pose_p[0] + eps, knife_pose_p[1], knife_pose_p[2]]
        box2_pos_q = [0, 0, 0, 1]

        self.box2_1 = create_box(
            scene=self,
            pose=sapien.Pose(box1_pos_p, box1_pos_q),
            half_size=(0.01, 0.05, 0.05),
            color=(0, 0, 0),
            name="box3",
            is_static=True,
        )
        self.box2_2 = create_box(
            scene=self,
            pose=sapien.Pose(box2_pos_p, box2_pos_q),
            half_size=(0.01, 0.05, 0.05),
            color=(0, 0, 0),
            name="box4",
            is_static=True,
        )

        self.knife_2 = create_actor(
            scene=self,
            pose=sapien.Pose(knife_pose_p, knife_pose_q),
            # pose=sapien.Pose([0, -0.06, 0.6875], [0, 0, 0.995, 0.105]),
            modelname="034_knife",
            convex=True,
            model_id=0,
            is_static=False,
        )
        self.knife_2.set_mass(0.0045) 
        
        # 打印刀具位置用于调试
        # print(f"knife_1 created at: [{knife_1_x:.3f}, {knife_1_y:.3f}, 0.773]")
        # print(f"knife_2 created at: [{knife_2_x:.3f}, {knife_2_y:.3f}, 0.773]") 

    def play_once(self, loop_times=6):
        # 强制启用路径规划以避免IndexError
        self.need_plan = True
        
        # self.wait(10)
        # print(self.board.get_pose().p)
        knife_1_pose = self.knife_1.get_pose().p
        knife_2_pose = self.knife_2.get_pose().p
        
        # 确定每把刀应该用哪只手抓取
        arm_tag_1 = ArmTag("left" if knife_1_pose[0] < 0 else "right")
        arm_tag_2 = ArmTag("left" if knife_2_pose[0] < 0 else "right")
        
        # print(f"knife_1 position: {knife_1_pose}, assigned to: {arm_tag_1}")
        # print(f"knife_2 position: {knife_2_pose}, assigned to: {arm_tag_2}")
        
        # 检查是否两把刀被分配给同一只手
        if arm_tag_1 == arm_tag_2:
            # print(f"Warning: Both knives assigned to {arm_tag_1} hand!")
            # 强制分配：knife_1给右手，knife_2给左手
            arm_tag_1 = ArmTag("right")
            arm_tag_2 = ArmTag("left")
            print(f"Reassigned: knife_1 -> {arm_tag_1}, knife_2 -> {arm_tag_2}")

        # 先抓取第一把刀
        self.move(self.grasp_actor(self.knife_1, arm_tag=arm_tag_1, pre_grasp_dis=0.12, grasp_dis=0.01))
        # 把第一把刀往上移动一点
        self.move(self.move_by_displacement(arm_tag_1, z=0.15, move_axis="world"))

        # 记录第一个手臂的位置（用于后面放回）
        arm_1_pos = np.array(self.get_arm_pose(arm_tag_1)[:3])

        # 再抓取第二把刀
        self.move(self.grasp_actor(self.knife_2, arm_tag=arm_tag_2, pre_grasp_dis=0.12, grasp_dis=0.01))
        # 把第二把刀往上移动一点
        self.move(self.move_by_displacement(arm_tag_2, z=0.15, move_axis="world"))

        # 记录第二个手臂的位置（用于后面放回）
        arm_2_pos = np.array(self.get_arm_pose(arm_tag_2)[:3])

        # 移动到砧板上方，左手左一点，右手右一点
        curr_pos_left = np.array(self.get_arm_pose(ArmTag("left"))[:3])
        target_pos_left = self.board_pos.p + np.array([-0.1, -0.1, 0.25])
        error_left = target_pos_left - curr_pos_left
        curr_pos_right = np.array(self.get_arm_pose(ArmTag("right"))[:3])
        target_pos_right = self.board_pos.p + np.array([0.1, -0.1, 0.25])
        error_right = target_pos_right - curr_pos_right
        
        # 使用双臂同时移动
        left_move = self.move_by_displacement(ArmTag("left"), x=error_left[0], y=error_left[1], z=error_left[2], move_axis="world")
        right_move = self.move_by_displacement(ArmTag("right"), x=error_right[0], y=error_right[1], z=error_right[2], move_axis="world")
        self.move(left_move, right_move)

        self.wait(1)

        standart_z = target_pos_left[2]  # 假设两只手臂的z坐标相同
        # 循环往下剁 - 交错动作，随机决定左手还是右手先开始
        left_first = random.choice([True, False])
        
        # 初始状态：两个刀都在上方，需要先让一个刀下去开始
        if left_first:
            # 左手先下
            first_down = self.move_by_displacement(ArmTag("left"), z=-0.1, move_axis="world")
            self.move(first_down)
            self.wait(0.15)
        else:
            # 右手先下
            first_down = self.move_by_displacement(ArmTag("right"), z=-0.1, move_axis="world")
            self.move(first_down)
            self.wait(0.15)
        
        # 交错循环
        for i in range(loop_times-1):
            if left_first:
                if i % 2 == 0:
                    # 左手上，右手下
                    left_up = self.move_by_displacement(ArmTag("left"), z=0.1, move_axis="world")
                    right_down = self.move_by_displacement(ArmTag("right"), z=-0.1, move_axis="world")
                    self.move(left_up, right_down)
                else:
                    # 左手下，右手上
                    left_down = self.move_by_displacement(ArmTag("left"), z=-0.1, move_axis="world")
                    right_up = self.move_by_displacement(ArmTag("right"), z=0.1, move_axis="world")
                    self.move(left_down, right_up)
            else:
                if i % 2 == 0:
                    # 右手上，左手下
                    right_up = self.move_by_displacement(ArmTag("right"), z=0.1, move_axis="world")
                    left_down = self.move_by_displacement(ArmTag("left"), z=-0.1, move_axis="world")
                    self.move(right_up, left_down)
                else:
                    # 右手下，左手上
                    right_down = self.move_by_displacement(ArmTag("right"), z=-0.1, move_axis="world")
                    left_up = self.move_by_displacement(ArmTag("left"), z=0.1, move_axis="world")
                    self.move(right_down, left_up)
            self.wait(0.3)
        
        self.wait(1)
        # 保证都提起到标准高度
        left_to_standard = self.move_by_displacement(ArmTag("left"), z=standart_z - self.get_arm_pose(ArmTag("left"))[2], move_axis="world")
        right_to_standard = self.move_by_displacement(ArmTag("right"), z=standart_z - self.get_arm_pose(ArmTag("right"))[2], move_axis="world")
        self.move(left_to_standard, right_to_standard)
        self.wait(0.5)

        # 放回去 - 双臂同时动作
        # 计算左手和右手分别要回到的位置
        curr_pos_left = np.array(self.get_arm_pose(ArmTag("left"))[:3])
        curr_pos_right = np.array(self.get_arm_pose(ArmTag("right"))[:3])
        
        # 判断哪只手拿的是哪把刀，基于之前记录的arm_tag
        if arm_tag_1 == "left":
            left_target_pos = arm_1_pos
            right_target_pos = arm_2_pos
        else:
            left_target_pos = arm_2_pos
            right_target_pos = arm_1_pos
        
        error_pos_left = left_target_pos - curr_pos_left
        error_pos_right = right_target_pos - curr_pos_right
        
        # 双臂同时移动到放置位置
        left_move = self.move_by_displacement(ArmTag("left"), x=error_pos_left[0], y=error_pos_left[1], z=error_pos_left[2], move_axis="world")
        right_move = self.move_by_displacement(ArmTag("right"), x=error_pos_right[0], y=error_pos_right[1], z=error_pos_right[2], move_axis="world")
        self.move(left_move, right_move)
        self.wait(0.5)
        
        # 双臂同时向下移动
        left_down = self.move_by_displacement(ArmTag("left"), z=-0.15, move_axis="world")
        right_down = self.move_by_displacement(ArmTag("right"), z=-0.15, move_axis="world")
        self.move(left_down, right_down)
        self.wait(0.5)
        
        # 双臂同时松开夹爪
        left_open = self.open_gripper(ArmTag("left"))
        right_open = self.open_gripper(ArmTag("right"))
        self.move(left_open, right_open)
        
        # 双臂同时向上移动
        left_up = self.move_by_displacement(ArmTag("left"), z=0.15, move_axis="world")
        right_up = self.move_by_displacement(ArmTag("right"), z=0.15, move_axis="world")
        self.move(left_up, right_up)

        self.wait(2)

        # print(self.board.get_pose().q)
        # 统一 info 输出格式：记录两把刀与对应手臂；{A}/{B} 为两把刀模型，占位符 {a}/{b} 为抓取它们的手臂
        # 保持与其它任务一致：self.info 在 _base_task 中初始化，这里只填充 info 字段
        if not hasattr(self, "info") or not isinstance(self.info, dict):
            self.info = {}

        # 根据先前确定的 arm_tag_1 / arm_tag_2 与 knife_1 / knife_2 绑定关系
        # arm_tag_1 抓取 self.knife_1, arm_tag_2 抓取 self.knife_2
        self.info["info"] = {
            "{A}": "034_knife/base0",  # 模型名称占位；如后续需要区分实例可在创建时记录 id
            "{B}": "034_knife/base0",  # 第二把同模型
            "{a}": str(arm_tag_1),      # 第一把刀对应手臂（left/right）
            "{b}": str(arm_tag_2),      # 第二把刀对应手臂
        }

        return self.info

    def check_success(self):
        return True

    def validate_data_dimensions(self, data_dict, location="未知位置"):
        """
        验证数据维度和类型的通用方法
        
        Args:
            data_dict: 包含数据的字典，键为数据名称，值为数据
            location: 调用位置的描述，用于错误信息
            
        Returns:
            tuple: (is_valid, error_msg)
        """
        try:
            for name, data in data_dict.items():
                # 检查数据是否为空
                if data is None:
                    error_msg = f"[{location}] {name} 数据为 None"
                    print(f"[Data Validation] {error_msg}")
                    return False, error_msg
                
                # 转换为numpy数组
                if not isinstance(data, np.ndarray):
                    try:
                        data = np.array(data)
                        print(f"[Data Validation] [{location}] {name} 已转换为numpy数组")
                    except Exception as e:
                        error_msg = f"[{location}] {name} 无法转换为numpy数组: {e}"
                        print(f"[Data Validation] {error_msg}")
                        return False, error_msg
                
                # 检查数据长度
                if len(data) == 0:
                    error_msg = f"[{location}] {name} 数据长度为0"
                    print(f"[Data Validation] {error_msg}")
                    return False, error_msg
                
                # 检查数据维度
                if data.ndim == 1:
                    print(f"[Data Validation] [{location}] {name} 为1维数据，形状: {data.shape}")
                    if len(data) < 3:
                        error_msg = f"[{location}] {name} 1维数据长度不足: {len(data)}"
                        print(f"[Data Validation] {error_msg}")
                        return False, error_msg
                elif data.ndim == 2:
                    print(f"[Data Validation] [{location}] {name} 为2维数据，形状: {data.shape}")
                    if data.shape[0] == 0 or data.shape[1] == 0:
                        error_msg = f"[{location}] {name} 2维数据尺寸无效: {data.shape}"
                        print(f"[Data Validation] {error_msg}")
                        return False, error_msg
                else:
                    error_msg = f"[{location}] {name} 数据维度异常: {data.ndim}，期望1维或2维"
                    print(f"[Data Validation] {error_msg}")
                    return False, error_msg
                
                # 检查数据类型
                if not np.issubdtype(data.dtype, np.number):
                    print(f"[Data Validation] [{location}] 警告: {name} 数据类型非数值: {data.dtype}")
                
                print(f"[Data Validation] [{location}] ✅ {name} 验证通过: 形状={data.shape}, 类型={data.dtype}")
            
            return True, "所有数据验证通过"
            
        except Exception as e:
            error_msg = f"[{location}] 数据验证过程中出现异常: {e}"
            print(f"[Data Validation] {error_msg}")
            return False, error_msg

    def record_loop_metric(self):
        left_arm_pos=self.get_arm_pose(arm_tag = ArmTag("left"))
        right_arm_pos=self.get_arm_pose(arm_tag = ArmTag("right"))
        knife_1_p = self.knife_1.get_pose().p
        knife_2_p = self.knife_2.get_pose().p
        # chopping_board_p = self.board.get_pose().p

        if "knife_1_pos" not in self.loop_metric:
            self.loop_metric["knife_1_pos"] = []
            self.loop_metric["knife_2_pos"] = []
            self.loop_metric["left_arm_pos"] = []
            self.loop_metric["right_arm_pos"] = []
            
        self.loop_metric["knife_1_pos"].append(knife_1_p)
        self.loop_metric["knife_2_pos"].append(knife_2_p)
        self.loop_metric["left_arm_pos"].append(left_arm_pos)
        self.loop_metric["right_arm_pos"].append(right_arm_pos)

    def analyze_loop_metric(self):
        try:
            # 保存到文件，方便后续调试
            np.save(f"{self.eval_video_path}/episode{self.test_num}.npz", self.loop_metric)
            
            from envs.utils.analyze_tools.peak_detect import peak_detect
            results = ""

            ############################ 获取并预处理数据 ############################
            
            knife1_pos = self.loop_metric['knife_1_pos'] # np数组, shape (N, 3)
            knife2_pos = self.loop_metric['knife_2_pos'] # np数组, shape (N, 3)
            left_arm_pos = self.loop_metric['left_arm_pos'] # np数组, shape (N, 3)
            right_arm_pos = self.loop_metric['right_arm_pos'] # np数组, shape (N, 3)

            # 先转换为numpy数组
            knife1_pos = np.array(knife1_pos)
            knife2_pos = np.array(knife2_pos)
            left_arm_pos = np.array(left_arm_pos)
            right_arm_pos = np.array(right_arm_pos)

            # 将x偏左的刀定义为刀1，偏右的刀定义为刀2
            # 同时重新排序对应的手臂位置数据
            if knife1_pos[0, 0] > knife2_pos[0, 0]:
                print(f"[Debug] 交换刀具和手臂数据：原knife1_x={knife1_pos[0, 0]:.3f} > 原knife2_x={knife2_pos[0, 0]:.3f}")
                knife1_pos, knife2_pos = knife2_pos, knife1_pos
                print(f"[Debug] 交换后：knife1_x={knife1_pos[0, 0]:.3f}, knife2_x={knife2_pos[0, 0]:.3f}")
            else:
                print(f"[Debug] 无需交换：knife1_x={knife1_pos[0, 0]:.3f} <= knife2_x={knife2_pos[0, 0]:.3f}")

            knife_1_init_x = knife1_pos[0, 0]
            knife_2_init_x = knife2_pos[0, 0]
            knife1_pos_z = knife1_pos[:, 2]
            knife2_pos_z = knife2_pos[:, 2]
            
            ############################ 获取并预处理数据 ##############################

            #################### 截取有效切菜部分 ####################
            # 截断开始部分：找到刀具z轴上升超过初始位置0.01的位置
            high_enough_index_1 = np.where(knife1_pos_z > knife1_pos_z[0] + 0.04)[0]
            start_idx_knife1 = high_enough_index_1[0] if len(high_enough_index_1) > 0 else 0
            high_enough_index_2 = np.where(knife2_pos_z > knife2_pos_z[0] + 0.04)[0]
            start_idx_knife2 = high_enough_index_2[0] if len(high_enough_index_2) > 0 else 0

            close_index_1 = np.where(knife1_pos[:, 0] > knife_1_init_x + 0.01)[0] # 找到刀1 x轴大于初始位置+0.01的位置
            close_index_2 = np.where(knife2_pos[:, 0] < knife_2_init_x - 0.01)[0] # 找到刀2 x轴小于初始位置-0.01的位置
            
            start_idx_knife1 = min(start_idx_knife1, close_index_1[0]) if len(close_index_1) > 0 else start_idx_knife1
            start_idx_knife2 = min(start_idx_knife2, close_index_2[0]) if len(close_index_2) > 0 else start_idx_knife2
                                          
            print(f"[Debug] start_idx_knife1={start_idx_knife1}, start_idx_knife2={start_idx_knife2}")

            knife1_pos = knife1_pos[start_idx_knife1:]
            knife2_pos = knife2_pos[start_idx_knife2:]
            left_arm_pos = left_arm_pos[start_idx_knife1:]
            right_arm_pos = right_arm_pos[start_idx_knife2:]

            index_1 = np.where(left_arm_pos[:, 2] > 1.05)[0]
            index_2 = np.where(right_arm_pos[:, 2] > 1.05)[0]
            if len(index_1) > 0:
                knife1_pos = knife1_pos[index_1[0]:]
                left_arm_pos = left_arm_pos[index_1[0]:]
            if len(index_2) > 0:
                knife2_pos = knife2_pos[index_2[0]:]
                right_arm_pos = right_arm_pos[index_2[0]:]
                
            index_1 = np.where(left_arm_pos[:, 2] < 1.01)[0]
            index_2 = np.where(right_arm_pos[:, 2] < 1.01)[0]
            if len(index_1) > 0:
                knife1_pos = knife1_pos[index_1[0]:]
                left_arm_pos = left_arm_pos[index_1[0]:]
            if len(index_2) > 0:
                knife2_pos = knife2_pos[index_2[0]:]
                right_arm_pos = right_arm_pos[index_2[0]:]
            
            print(f"[Debug] 切片后手臂数据形状: left_arm_pos={left_arm_pos.shape}, right_arm_pos={right_arm_pos.shape}")
            
            ############################### 截取有效切菜部分 ##############################
            # 重新计算 where_start_1 和 where_start_2 相对于新数组的索引
            
            delta_z_left = left_arm_pos[:, 2] - knife1_pos[:, 2]
            delta_z_right = right_arm_pos[:, 2] - knife2_pos[:, 2]

            # 估计左手和左刀的z轴高度关系
            exp_delta_z_left = np.median(delta_z_left[:20])
            exp_delta_z_right = np.median(delta_z_right[:20])
            print(f"[Debug] 估计的切菜时手臂与刀具z轴高度差：左手 {exp_delta_z_left:.4f}，右手 {exp_delta_z_right:.4f}")

            des_knife1_x= knife1_pos[0,0]
            des_knife2_x= knife2_pos[0,0]
            
            # 截断 
            end_idx_knife1 = np.where(knife1_pos[:,0] < (knife_1_init_x + des_knife1_x)/2)[0]
            end_idx_knife1 = end_idx_knife1[0] if len(end_idx_knife1) > 0 else len(knife1_pos[:,0]) - 1
            end_idx_knife2 = np.where(knife2_pos[:,0] > (knife_2_init_x + des_knife2_x)/2)[0]
            end_idx_knife2 = end_idx_knife2[0] if len(end_idx_knife2) > 0 else len(knife2_pos[:,0]) - 1

            where_left = np.where(delta_z_left > exp_delta_z_left + 0.05)[0]
            end_idx_left = where_left[0] if len(where_left) > 0 else len(knife1_pos[:,0]) - 1
            where_right = np.where(delta_z_right > exp_delta_z_right + 0.05)[0]
            end_idx_right = where_right[0] if len(where_right) > 0 else len(knife2_pos[:,0]) - 1

            cut_end_idx_1 = min(end_idx_knife1, end_idx_left)
            cut_end_idx_2 = min(end_idx_knife2, end_idx_right)
            print(end_idx_knife1, end_idx_knife2)
            print(end_idx_left, end_idx_right)
            print(f"[Debug] 切菜结束索引：knife1={cut_end_idx_1}, knife2={cut_end_idx_2}")

            knife1_pos = knife1_pos[:cut_end_idx_1]
            knife2_pos = knife2_pos[:cut_end_idx_2]
            left_arm_pos = left_arm_pos[:cut_end_idx_1]
            right_arm_pos = right_arm_pos[:cut_end_idx_2]

            #################### 截取有效切菜部分 ####################

            ######################### 分析 ##########################

            knife1_pos_z = -knife1_pos[:, 2]
            knife2_pos_z = -knife2_pos[:, 2]
            
            left_arm_pos_z_raw = -left_arm_pos[:, 2]   # 只取z坐标
            right_arm_pos_z_raw = -right_arm_pos[:, 2]  # 只取z坐标
            
            # 参考PID控制思想减少抖动的滤波函数
            def pid_inspired_filter(signal, kp=0.7, kd=0.2, ki=0.1):
                """
                参考PID控制的滤波方法减少抖动
                kp: 比例项系数 (当前值的权重)
                kd: 微分项系数 (变化率的权重，用于减少快速抖动)  
                ki: 积分项系数 (历史趋势的权重，用于保持平滑)
                """
                if len(signal) < 2:
                    return signal
                    
                filtered_signal = np.zeros_like(signal)
                filtered_signal[0] = signal[0]
                
                integral = 0
                prev_error = 0
                
                for i in range(1, len(signal)):
                    # 当前误差（相对于滤波后的前一值）
                    error = signal[i] - filtered_signal[i-1]
                    
                    # 积分项（累积误差趋势）
                    integral += error
                    
                    # 微分项（误差变化率）
                    derivative = error - prev_error
                    
                    # PID输出（但这里是用于信号重构）
                    output = kp * error + ki * integral * 0.01 + kd * derivative
                    
                    # 更新滤波信号
                    filtered_signal[i] = filtered_signal[i-1] + output
                    prev_error = error
                    
                return filtered_signal
            
            # 应用PID风格的滤波减少抖动
            left_arm_pos_z = pid_inspired_filter(left_arm_pos_z_raw, kp=0.7, kd=0.2, ki=0.1)
            right_arm_pos_z = pid_inspired_filter(right_arm_pos_z_raw, kp=0.7, kd=0.2, ki=0.1)
            
            # 检查数据中大于-0.95的值够不够多
            index_1=np.where(left_arm_pos_z > -0.98)[0]
            index_2=np.where(right_arm_pos_z > -0.98)[0]
            if len(index_1)/len(left_arm_pos_z) < 0.1:
                left_arm_pos_z=left_arm_pos_z[0:0]
            if len(index_2)/len(right_arm_pos_z) < 0.1:
                right_arm_pos_z=right_arm_pos_z[0:0]

            ########## peak detect ##########
            num_peaks_knife1, peak_positions_knife1 = peak_detect(
                left_arm_pos_z, 
                smooth=True,
                smooth_window=23,
                height_factor=0.25,
                distance_factor=18,
                prominence_factor=0.02,
                save_plot=True, 
                save_path=f"{self.eval_video_path}/episode{self.test_num}_knife1.png"
            )
            num_peaks_knife2, peak_positions_knife2 = peak_detect(
                right_arm_pos_z, 
                smooth=True,
                smooth_window=23,
                height_factor=0.25,
                distance_factor=18,
                prominence_factor=0.02,
                save_plot=True, 
                save_path=f"{self.eval_video_path}/episode{self.test_num}_knife2.png"
            )

            loop_times = num_peaks_knife1 + num_peaks_knife2
            results += f"🔪 刀1往下次数：{num_peaks_knife1} 次；刀2往下次数：{num_peaks_knife2} 次；总切菜次数：{loop_times} 次。\n"

            all_peak_positions = np.sort(np.concatenate((np.array(peak_positions_knife1), np.array(peak_positions_knife2))))
            results += f"⬇️ 切菜动作发生的时间点（帧数）：{all_peak_positions.tolist()}\n"

            loop_info = {
                "loop_times": loop_times,
                "gap_times": np.diff(all_peak_positions).tolist() if len(all_peak_positions) > 1 else [],
                "peak_positions": all_peak_positions.tolist()
            }

        except Exception as e:
            print(f"[Loop Analysis] 分析过程中出现错误: {e}")
            results += f"❌ 分析过程中出现错误: {e}\n"
            loop_info = {
                "loop_times": 0,
                "gap_times": [],
                "peak_positions": [],
                "error_msg": str(e)
            }

            
        file_path = f"{self.eval_video_path}/episode{self.test_num}_loop_info.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for key, value in loop_info.items():
                f.write(f"{key}: {value}\n")
        return loop_info
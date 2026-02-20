from ._base_task import Base_Task
from .utils import *
import sapien
import math
from transforms3d.euler import quat2euler, euler2quat

from termcolor import cprint


class cut_carrot_knife(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.loop_counter = 0
        
        # ===== 状态机式接触检测参数 =====
        self.contact_state = False  # 当前接触状态：True=接触中, False=非接触
        self.contact_frames = 0  # 连续接触/非接触帧计数器
        self.contact_state_threshold = 5  # 连续N帧才能切换状态（防抖）
        
        self.cut_count = 0  # 切割次数（状态切换计数）
        self.cut_frames = []  # 每次切割发生的帧号列表
        self.gap_times = []  # 记录每次切割之间的间隔帧数
        
        self.metric_frame_counter = 0  # 内部帧计数器
        self.first_cut = False  # 是否已经开始切割
        
        # 用于调试的接触历史
        self.contact_history = []  # (raw_contact, state)

    def load_knife(self, is_random=False):
        if is_random:
            knife_pose_p = [
                np.random.uniform(0.10, 0.25),
                np.random.uniform(0.10, 0.25),
                0.77301395,
            ]
        else:
            knife_pose_p = [0.15, 0.15, 0.77301395]

        self.knife_init_p = knife_pose_p.copy()
        knife_pose_q = [0, 0, 1, 0]

        eps = 0.03
        box1_pos_p = [knife_pose_p[0] - eps, knife_pose_p[1], knife_pose_p[2]]
        box1_pos_q = [0, 0, 0, 1]
        box2_pos_p = [knife_pose_p[0] + eps, knife_pose_p[1], knife_pose_p[2]]
        box2_pos_q = [0, 0, 0, 1]

        self.box1 = create_box(
            scene=self,
            pose=sapien.Pose(box1_pos_p, box1_pos_q),
            half_size=(0.01, 0.05, 0.05),
            color=(0, 0, 0),
            name="box1",
            is_static=True,
        )
        self.box2 = create_box(
            scene=self,
            pose=sapien.Pose(box2_pos_p, box2_pos_q),
            half_size=(0.01, 0.05, 0.05),
            color=(0, 0, 0),
            name="box2",
            is_static=True,
        )

        self.knife = create_actor(
            scene=self,
            pose=sapien.Pose(knife_pose_p, knife_pose_q),
            # pose=sapien.Pose([0, -0.06, 0.6875], [0, 0, 0.995, 0.105]),
            modelname="034_knife",
            convex=True,
            model_id=0,
            is_static=False,
        )
        self.knife.set_mass(0.0045)

        # Register key objects for 6D pose tracking
        self.set_key_objects({"knife": self.knife})

    def load_carrot(self, is_random=False):
        if is_random:
            carrot_pose_p = [
                np.random.uniform(0, 0.10),
                np.random.uniform(-0.2, 0),
                0.779127,
            ]
        else:
            carrot_pose_p = [0.05, -0.1, 0.779127]

        self.carrot_init_p = carrot_pose_p.copy()
        self.carrot_pose = [carrot_pose_p, [0.686108, 0.156719, -0.59445, 0.389003]]
        self.carrot = create_actor(
            scene=self,
            pose=sapien.Pose(self.carrot_pose[0], self.carrot_pose[1]),
            modelname="135_carrot",
            convex=True,
            model_id=0,
            is_static=True,
        )
        self.carrot.set_mass(0.025)

    def load_actors(self):
        is_random = True
        self.load_knife(is_random)
        self.load_carrot(is_random)        

    def play_once(self, loop_times=6):
        # self.wait(10)
        # print(self.carrot.get_pose())
        # 获取刀的位置
        knife_pose = self.knife.get_pose().p
        # # 根据刀的位置选择左手或右手
        arm_tag = ArmTag("left" if knife_pose[0] < 0 else "right")

        # 用选定的手臂抓取刀
        self.move(self.grasp_actor(self.knife, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01))
        # 把刀往上移动一点
        self.move(self.move_by_displacement(arm_tag, z=0.15, move_axis="arm"))

        # 夹爪向下
        target_quat = euler2quat(0, math.pi/2, 0)
        # 移动到胡萝卜的pose
        curr_pos = np.array(self.get_arm_pose(ArmTag("left"))[:3])
        target_pos = np.array(self.carrot.get_pose().p)
        target_pos[0] -= 0.15
        error_pos = target_pos - curr_pos

        self.move(self.move_by_displacement(ArmTag("left"), x=error_pos[0], y=error_pos[1], z=error_pos[2], quat=target_quat, move_axis="world"))

        self.wait(1)

        # 刀切胡萝卜
        target_pos[0] += 0.23
        target_pos[1] -= 0.2
        target_pos[2] += 0.3
        curr_pos = np.array(self.get_arm_pose(arm_tag)[:3])
        error_pos = target_pos - curr_pos
        self.move(self.move_by_displacement(arm_tag, x=error_pos[0], y=error_pos[1], z=error_pos[2], move_axis="world"))

        # cut for {loop_times} times
        # 总共切0.125m，每次切完后往左移动一点
        total_left_dis = 0.125
        # 一次移动的距离
        if loop_times > 1:
            left_dis_per_time = total_left_dis / (loop_times - 1)
        else:
            left_dis_per_time = 0.075
        for i in range(loop_times):
            if loop_times == 1:
                self.move(self.move_by_displacement(arm_tag, x=-left_dis_per_time, move_axis="world"))
            self.move(self.move_by_displacement(arm_tag, z=-0.1, move_axis="arm"))
            self.wait(0.2)
            self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))
            self.wait(0.5)
            if i != loop_times - 1:
                self.move(self.move_by_displacement(arm_tag, x=-left_dis_per_time, move_axis="world"))

                self.loop_counter += 1

        self.wait(1)

        # 放回去
        target_pos = curr_pos
        
        curr_pos = np.array(self.get_arm_pose(arm_tag)[:3])
        error_pos = target_pos - curr_pos
        self.move(self.move_by_displacement(arm_tag, x=error_pos[0], y=error_pos[1], z=error_pos[2], move_axis="world"))
        self.wait(0.5)
        self.move(self.move_by_displacement(arm_tag, z=-0.15, move_axis="arm"))
        self.wait(0.5)
        self.move(self.open_gripper(arm_tag))
        self.move(self.move_by_displacement(arm_tag, z=0.15, move_axis="arm"))

        self.wait(2)

        self.info["info"] = {"{A}": "034_knife/base0", "{a}": str(arm_tag)}
        return self.info

    def check_success(self):
        # 如果刀的z坐标低于0.5，或者高于1.2，或x小于-0.1 则认为失败
        # 如果右手的x左边小于左手的x则认为失败
        knife_pose = self.knife.get_pose().p
        if knife_pose[2] < 0.5 \
            or knife_pose[2] > 1.2 \
            or knife_pose[0] < -0.1 \
            or self.get_arm_pose(ArmTag("right"))[0] < self.get_arm_pose(ArmTag("left"))[0]:
            return False
        return True
    
    ################################################################################

    ########################## 以下为检测切割次数的代码 ##############################
    
    ################################################################################



    def update_contact_state(self):
        """
        更新接触状态机
        使用状态机：连续N帧接触->切换到"接触状态"，连续N帧非接触->切换到"非接触状态"
        状态从 False->True 时计数一次切割
        """
        knife_p = self.knife.get_pose().p
        carrot_p = self.carrot.get_pose().p
        
        # 检测物理接触
        is_contact_raw = self.check_actors_contact(self.knife.get_name(), self.carrot.get_name())
        
        # 额外条件：刀要在胡萝卜附近（y方向距离小于阈值）
        is_near = abs(knife_p[1] - carrot_p[1]) < 0.15
        
        # 综合判断：物理接触 + 位置接近
        is_contact = is_contact_raw and is_near
        
        # 记录原始接触和当前状态（用于调试）
        self.contact_history.append((is_contact, self.contact_state))
        
        # 状态转移逻辑
        if is_contact:
            # 当前帧有接触
            if self.contact_state:
                # 已在接触状态，保持不变，计数器清零
                self.contact_frames = 0
            else:
                # 在非接触状态，累计接触帧数
                self.contact_frames += 1
                if self.contact_frames >= self.contact_state_threshold:
                    # 达到阈值，切换到接触状态，并计数一次切割
                    self.contact_state = True
                    self.cut_count += 1
                    self.cut_frames.append(self.metric_frame_counter)
                    
                    # 计算间隔
                    if len(self.cut_frames) > 1:
                        gap = self.cut_frames[-1] - self.cut_frames[-2]
                        self.gap_times.append(gap)
                    
                    self.contact_frames = 0
                    print(f"🔪 切割事件 #{self.cut_count} (帧: {self.metric_frame_counter}, 刀X: {knife_p[0]:.3f})")
                    
                    if not self.first_cut:
                        self.first_cut = True
                        print(f">>> 首次接触胡萝卜")
        else:
            # 当前帧无接触
            if not self.contact_state:
                # 已在非接触状态，保持不变，计数器清零
                self.contact_frames = 0
            else:
                # 在接触状态，累计非接触帧数
                self.contact_frames += 1
                if self.contact_frames >= self.contact_state_threshold:
                    # 达到阈值，切换到非接触状态
                    self.contact_state = False
                    self.contact_frames = 0
                    # print(f"  [状态切换] 离开接触 (帧: {self.metric_frame_counter})")
    
    def record_loop_metric(self):
        """
        使用状态机式接触检测来统计切割次数
        同时记录位置信息用于左移距离统计和可视化
        """
        # 使用内部帧计数器
        current_frame = self.metric_frame_counter
        
        knife_p = self.knife.get_pose().p
        carrot_p = self.carrot.get_pose().p

        left_arm = self.get_arm_pose(ArmTag("left"))
        right_arm = self.get_arm_pose(ArmTag("right"))
        
        # 判断任务是否结束（刀回到初始区域或异常位置）
        if knife_p[2] < 0.5 or knife_p[2] > 1.2 or knife_p[0] < -0.1:
            return
        
        # 初始化记录
        if "cut_events" not in self.loop_metric:
            self.loop_metric["cut_events"] = []  # 记录每次切割事件的帧数
            self.loop_metric["knife_pos"] = []  # 刀的位置（用于左移距离统计和可视化）
            self.loop_metric["carrot_pos"] = []  # 胡萝卜的位置
            self.loop_metric["contact_state"] = []  # 记录每帧的接触状态
            self.loop_metric["knife_x"] = []  # 刀的X坐标（用于左移分析）

            self.loop_metric["left_arm"] = []
            self.loop_metric["right_arm"] = []
        
        # 更新状态机
        self.update_contact_state()
        
        # 记录位置信息和状态
        self.loop_metric["knife_pos"].append(knife_p.copy())
        self.loop_metric["carrot_pos"].append(carrot_p.copy())
        self.loop_metric["knife_x"].append(knife_p[0])
        self.loop_metric["contact_state"].append(self.contact_state)
        self.loop_metric["left_arm"].append(left_arm)
        self.loop_metric["right_arm"].append(right_arm)
        
        # 递增帧计数器
        self.metric_frame_counter += 1

    def analyze_loop_metric(self):
        """
        使用状态机式接触检测结果来分析切割次数，同时保留左移距离的统计
        """

        debug = True

        try:
            # 保存到文件，方便后续调试
            np.save(f"{self.eval_video_path}/episode{self.test_num}.npz", self.loop_metric)

            # 检查是否有切割事件记录
            if "cut_events" not in self.loop_metric:
                print(f"[Loop Metric] 未记录切割事件数据")
                return {
                    "loop_times": 0,
                    "gap_times": [],
                    "cut_frames": [],
                    "left_move_distance": 0.0,
                    "method": "state_machine_contact_detection"
                }
            
            # 使用状态机检测的结果
            loop_times = self.cut_count
            gap_times = self.gap_times.copy()
            cut_events = self.cut_frames.copy()
            
            # 计算刀的左移距离
            left_move_distance = 0.0
            if "knife_x" in self.loop_metric and len(self.loop_metric["knife_x"]) > 0:
                knife_x = np.array(self.loop_metric["knife_x"])
                
                # 找到切割事件对应的X坐标
                if len(cut_events) > 1:
                    # 从第一次切割到最后一次切割的X轴位移
                    start_idx = cut_events[0]
                    end_idx = cut_events[-1]
                    
                    if start_idx < len(knife_x) and end_idx < len(knife_x):
                        start_x = knife_x[start_idx]
                        end_x = knife_x[end_idx]
                        # 负值表示向左移动(X减小)
                        left_move_distance = start_x - end_x
            
            print(f"[Loop Analysis] 基于状态机接触检测:")
            print(f"  - 检测到切割次数: {loop_times}")
            print(f"  - 切割发生在帧: {cut_events}")
            print(f"  - 间隔帧数: {gap_times}")
            print(f"  - 刀向左移动距离(实际/设计): {left_move_distance:.3f}m / {0.125}m")
            
            collision_loop_times = loop_times
            collision_loop_cut_frames = cut_events

            peak_loop_times = None
            peak_loop_cut_frames = None
            
            # 可视化：绘制接触状态和Z轴波形
            if "knife_pos" in self.loop_metric and len(self.loop_metric["knife_pos"]) > 10:
                import matplotlib.pyplot as plt
                from .utils.analyze_tools.peak_detect import peak_detect
                
                # 将列表转换为numpy数组
                knife_pos = np.array(self.loop_metric["knife_pos"])  # shape (N, 3)
                contact_state = np.array(self.loop_metric.get("contact_state", []))
                
                # 先检查：刀是否掉在了桌面或者地上
                if np.any(knife_pos[:, 2] < 0.5):
                    print("[可视化] 警告：刀在任务期间掉落")
                    raise ValueError("刀在任务期间掉落，无法进行有效分析")

                # 使用基于刀与胡萝卜 y 比较的截取逻辑，然后对截取段的 z 轴做峰值检测
                # 逻辑：只保留那些刀的 y 小于胡萝卜的 y 的帧（刀在胡萝卜下方/接近位置）
                carrot_pos = np.array(self.loop_metric["carrot_pos"])  # shape (N, 3)

                # 构建 mask：刀的 y < 胡萝卜的 y
                try:
                    mask = knife_pos[:, 1] < carrot_pos[:, 1]
                except Exception:
                    mask = np.zeros(len(knife_pos), dtype=bool)

                knife_pos_filtered = knife_pos[mask]
                interval_length = len(knife_pos_filtered)

                if interval_length > 5:
                    # 取反的 z 轴用于峰值检测（下压为峰）
                    knife_z_negative = -knife_pos_filtered[:, 2]
                    num_peaks, peak_positions = peak_detect(
                        knife_z_negative,
                        smooth=True,
                        smooth_window=20,
                        height_factor=0.15,
                        distance_factor=30,
                        prominence_factor=0.04,
                        save_plot=True,
                        save_path=f"{self.eval_video_path}/episode{self.test_num}.png",
                    )

                    # 将局部峰位置映射回全序列帧索引（可选）
                    # 找到原序列中被保留帧的索引
                    original_indices = np.nonzero(mask)[0]
                    peak_positions_global = [int(original_indices[p]) for p in peak_positions]

                    peak_loop_times = num_peaks
                    peak_loop_cut_frames = peak_positions_global

                    print(f"📊 参考（y-filtered）峰值检测检测到 {num_peaks} 个峰值，帧位置(全局): {peak_positions_global}")
                else:
                    print(f"📊 参考峰值检测：有效截取段长度太短（{interval_length}），跳过峰值检测")

            ### summarize loop_info
            if peak_loop_times is not None:
                if collision_loop_times == peak_loop_times:
                    loop_info = {
                        "loop_times": collision_loop_times,
                        "cut_frames": collision_loop_cut_frames,
                        "gap_times": gap_times,
                        "left_move_distance": float(left_move_distance),
                        "supplement": "two method agree",
                    }
                elif collision_loop_times < peak_loop_times:
                    loop_info = {
                        "loop_times": peak_loop_times,
                        "cut_frames": peak_loop_cut_frames,
                        "gap_times": gap_times,
                        "left_move_distance": float(left_move_distance),
                        "supplement": "可能有的没切到",
                    }
                else:
                    loop_info = {
                        "loop_times": collision_loop_times,
                        "cut_frames": collision_loop_cut_frames,
                        "gap_times": gap_times,
                        "left_move_distance": float(left_move_distance),
                        "supplement": "the peak detection may miss some cuts",
                    }
            else:
                loop_info = {
                    "loop_times": collision_loop_times,
                    "cut_frames": collision_loop_cut_frames,
                    "gap_times": gap_times,
                    "left_move_distance": float(left_move_distance),
                    "method": "state_machine_contact_detection",
                }
            
            if debug:
                # print("Loop Info:", loop_info)
                # 更规整的打印
                cprint("\n===== Loop Analysis Result =====", "cyan", attrs=["bold"])
                cprint(f"碰撞检测切割次数: {collision_loop_times}", "yellow")
                cprint(f"峰值检测切割次数: {peak_loop_times}", "yellow")
                cprint(f"左手移动距离: {left_move_distance}", "yellow")

                if "supplement" in loop_info:
                    cprint(f"补充说明: {loop_info['supplement']}", "magenta")

                cprint("================================\n", "cyan", attrs=["bold"])
        except Exception as e:
            print(f"[Loop Analysis] 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            loop_info = {
                "loop_times": -1,
                "gap_times": [],
                "contact_frames": [],
                "left_move_distance": 0.0,
                "method": "collision_detection",
                "error_msg": str(e)
            }

            
        file_path = f"{self.eval_video_path}/episode{self.test_num}_loop_info.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for key, value in loop_info.items():
                f.write(f"{key}: {value}\n")
        return loop_info
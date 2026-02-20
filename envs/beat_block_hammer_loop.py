from ._base_task import Base_Task
from .utils import *
import sapien
from ._GLOBAL_CONFIGS import *
import json


class beat_block_hammer_loop(Base_Task):

    def setup_demo(self, **kwags):
        # Optional: override object poses (e.g., from DemoGen pointcloud mapping)
        # Expected format:
        #   {
        #     "hammer": {"p": [x,y,z], "q": [qw,qx,qy,qz]}  # or list len 3/7
        #     "block":  {"p": [x,y,z], "q": [qw,qx,qy,qz]}
        #   }
        self.override_object_poses = kwags.get("override_object_poses", None)
        super()._init_task_env_(**kwags)
        self.end = False
        self.first_hit = False
        self.loop_counter = 0
        
        # ===== 状态机式接触检测参数 =====
        self.contact_state = False  # 当前接触状态：True=接触中, False=非接触
        self.contact_frames = 0  # 连续接触/非接触帧计数器
        self.contact_state_threshold = 2  # 连续N帧才能切换状态（防抖）
        
        self.hit_count = 0  # 敲击次数（状态切换计数）
        self.hit_frames = []  # 每次敲击发生的帧号列表
        self.gap_times = []  # 记录每次敲击之间的间隔帧数
        
        self.metric_frame_counter = 0  # 内部帧计数器
        
        # 用于调试的接触历史
        self.contact_history = []  # (raw_contact, state)

    def load_actors(self):
        def _as_pose(obj_pose, default_q):
            if obj_pose is None:
                return None
            # Accept dict or list/tuple
            if isinstance(obj_pose, dict):
                p = obj_pose.get("p", None)
                q = obj_pose.get("q", None)
                if p is None:
                    return None
                if q is None:
                    q = default_q
                return sapien.Pose(p, q)
            if isinstance(obj_pose, (list, tuple, np.ndarray)):
                arr = np.asarray(obj_pose, dtype=np.float64).reshape(-1)
                if arr.size == 3:
                    return sapien.Pose(arr.tolist(), default_q)
                if arr.size >= 7:
                    return sapien.Pose(arr[:3].tolist(), arr[3:7].tolist())
            return None

        override = self.override_object_poses or {}
        default_hammer_q = [0, 0, 0.995, 0.105]

        # 固定锤子位置
        hammer_pose = _as_pose(override.get("hammer", None), default_hammer_q)
        if hammer_pose is None:
            hammer_pose = sapien.Pose([0, -0.06, 0.783], default_hammer_q)
        self.hammer = create_actor(
            scene=self,
            pose=hammer_pose,
            modelname="020_hammer",
            convex=True,
            model_id=0,
        )
        
        # 原来的随机位置生成代码（已注释）
        block_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.05, 0.15],
            zlim=[0.76],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.5],
        )
        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2], 2)) < 0.001:
            block_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.05, 0.15],
                zlim=[0.76],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.5],
            )

        # 固定方块位置
        # block_pose = sapien.Pose([-0.20, 0.05, 0.76], [1, 0, 0, 0])
        block_override_pose = _as_pose(override.get("block", None), [1, 0, 0, 0])
        if block_override_pose is not None:
            block_pose = block_override_pose

        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.025, 0.025, 0.025),
            color=(1, 0, 0),
            name="box",
            is_static=True,
        )
        self.hammer.set_mass(0.001)

        self.add_prohibit_area(self.hammer, padding=0.10)
        self.prohibited_area.append([
            block_pose.p[0] - 0.05,
            block_pose.p[1] - 0.05,
            block_pose.p[0] + 0.05,
            block_pose.p[1] + 0.05,
        ])

        # Register key objects for 6D pose tracking
        self.set_key_objects({"hammer": self.hammer})

    def play_once(self, loop_times=3):
        # Get the position of the block's functional point return (p(x, y, z), 四元数(qw, qx, qy, qz))
        block_pose = self.block.get_functional_point(0, "pose").p
        # Determine which arm to use based on block position (left if block is on left side, else right)
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        # Grasp the hammer with the selected arm
        self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01))
        # Move the hammer upwards
        self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))

        # Perform beating action 3 times
        for i in range(loop_times):
            # Place the hammer on the block's functional point (position 1)
            self.move(
                self.place_actor(
                    self.hammer,
                    target_pose=self.block.get_functional_point(1, "pose"),
                    arm_tag=arm_tag,
                    functional_point_id=0,
                    pre_dis=0.06,
                    dis=0,
                    is_open=False,
                ))
            
            # Lift the hammer slightly after each beat (except the last one)
            if i < 2:
                self.move(self.move_by_displacement(arm_tag, z=0.03, move_axis="arm"))

            self.loop_counter += 1

                # return to the center of the table
        self.move(self.back_to_origin(arm_tag=arm_tag))
        
        # 在初始位置停留一小段时间
        self.wait(0.5)
        
        self.info["info"] = {"{A}": "020_hammer/base0", "{a}": str(arm_tag)}
        return self.info

    def check_success(self):
        # 如果锤子z轴位置太高或太低，说明出问题了，返回失败
        hammer_pose = self.hammer.get_pose().p
        if hammer_pose[2] > 1.2 or hammer_pose[2] < 0.5:
            return False
        return True
    
    def get_hit_state(self):
        """检测锤子是否击中方块（位置对齐 + 物理接触）"""
        hammer_target_pose = self.hammer.get_functional_point(0, "pose").p
        block_pose = self.block.get_functional_point(1, "pose").p
        eps = np.array([0.05, 0.05])
        
        # 检测位置对准和物理接触
        is_aligned = np.all(abs(hammer_target_pose[:2] - block_pose[:2]) < eps)
        is_contact = self.check_actors_contact(self.hammer.get_name(), self.block.get_name())
        
        return is_aligned and is_contact
    
    def update_contact_state(self):
        """
        更新接触状态机
        使用状态机：连续N帧接触->切换到"接触状态"，连续N帧非接触->切换到"非接触状态"
        状态从 False->True 时计数一次敲击
        """
        # 使用 get_hit_state 检测物理接触
        is_contact = self.get_hit_state()
        
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
                    # 达到阈值，切换到接触状态，并计数一次敲击
                    self.contact_state = True
                    self.hit_count += 1
                    self.hit_frames.append(self.metric_frame_counter)
                    
                    # 计算间隔
                    if len(self.hit_frames) > 1:
                        gap = self.hit_frames[-1] - self.hit_frames[-2]
                        self.gap_times.append(gap)
                    
                    self.contact_frames = 0
                    
                    hammer_p = self.hammer.get_pose().p
                    print(f"🔨 敲击事件 #{self.hit_count} (帧: {self.metric_frame_counter}, 锤子 Z: {hammer_p[2]:.3f})")
                    
                    if not self.first_hit:
                        self.first_hit = True
                        print(f">>> 首次接触方块")
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
    
    def record_loop_metric(self):
        """
        使用状态机式接触检测来统计敲击次数
        同时记录位置信息用于可视化
        """
        # 使用内部帧计数器
        current_frame = self.metric_frame_counter
        
        hammer_target_pose = self.hammer.get_functional_point(0, "pose").p
        block_pose = self.block.get_pose().p

        left_arm = self.get_arm_pose(ArmTag("left"))
        right_arm = self.get_arm_pose(ArmTag("right"))

        # 判断任务是否结束
        if hammer_target_pose[1] < -0.08 or hammer_target_pose[2] > 0.92:
            self.end = True
        
        if self.end:
            return
        
        # 初始化记录
        if "hit_events" not in self.loop_metric:
            self.loop_metric["hit_events"] = []  # 记录每次敲击事件的帧数
            self.loop_metric["hammer_pos"] = []  # 锤子的位置（用于可视化）
            self.loop_metric["hammer_pos_z"] = []  # z 轴位置用于可视化
            self.loop_metric["contact_state"] = []  # 记录每帧的接触状态
            self.loop_metric["left_arm"] = []  # 左臂位置
            self.loop_metric["right_arm"] = []  # 右臂位置
        
        # 更新状态机
        self.update_contact_state()
        
        # 记录位置信息和状态
        self.loop_metric["hammer_pos"].append(hammer_target_pose.copy())
        self.loop_metric["hammer_pos_z"].append(hammer_target_pose[2])
        self.loop_metric["contact_state"].append(self.contact_state)
        self.loop_metric["left_arm"].append(left_arm)
        self.loop_metric["right_arm"].append(right_arm)
        
        # 递增帧计数器
        self.metric_frame_counter += 1

    def analyze_loop_metric(self):
        """
        使用状态机式接触检测结果来分析敲击次数，同时保留峰值检测作为参考
        """
        from termcolor import cprint
        
        debug = True
        
        try:
            # 保存到文件，方便后续调试
            np.save(f"{self.eval_video_path}/episode{self.test_num}.npz", self.loop_metric)

            # 检查是否有敲击事件记录
            if "hit_events" not in self.loop_metric:
                print(f"[Loop Metric] 未记录敲击事件数据")
                return {
                    "loop_times": 0,
                    "gap_times": [],
                    "hit_frames": [],
                    "method": "state_machine_contact_detection"
                }

            
            # 使用状态机检测的结果
            collision_loop_times = self.hit_count
            collision_gap_times = self.gap_times.copy()
            collision_hit_frames = self.hit_frames.copy()
            
            print(f"[Loop Analysis] 基于状态机接触检测:")
            print(f"  - 检测到敲击次数: {collision_loop_times}")
            print(f"  - 敲击发生在帧: {collision_hit_frames}")
            print(f"  - 间隔帧数: {collision_gap_times}")
            
            peak_loop_times = None
            peak_hit_frames = None
            
            # 峰值检测作为参考
            if "hammer_pos_z" in self.loop_metric and len(self.loop_metric["hammer_pos_z"]) > 10:
                try:
                    from .utils.analyze_tools.peak_detect import peak_detect
                    
                    hammer_pos = np.array(self.loop_metric["hammer_pos"])
                    
                    # 先检查：锤子是否掉落
                    if np.any(hammer_pos[:, 2] < 0.5):
                        print("[可视化] 警告：锤子在任务期间掉落")
                        raise ValueError("锤子在任务期间掉落，无法进行有效分析")
                    
                    hammer_pos_z = -np.array(self.loop_metric["hammer_pos_z"])
                    
                    # 绘制图表用于可视化和峰值检测
                    num_peaks, peak_positions = peak_detect(
                        hammer_pos_z,
                        smooth=True,
                        smooth_window=15,
                        height_factor=0.2,
                        distance_factor=30,
                        prominence_factor=0.04,
                        save_plot=True,
                        save_path=f"{self.eval_video_path}/episode{self.test_num}.png"
                    )
                    
                    peak_loop_times = num_peaks
                    peak_hit_frames = peak_positions
                    
                    print(f"📊 参考峰值检测检测到 {num_peaks} 个峰值，帧位置: {peak_positions}")
                    
                except Exception as viz_error:
                    print(f"📊 参考峰值检测失败: {viz_error}")
                    peak_loop_times = None
            
            ### summarize loop_info
            if peak_loop_times is not None:
                if collision_loop_times == peak_loop_times:
                    loop_info = {
                        "loop_times": collision_loop_times,
                        "hit_frames": collision_hit_frames,
                        "gap_times": collision_gap_times,
                        "supplement": "两种方法一致",
                    }
                elif collision_loop_times < peak_loop_times:
                    loop_info = {
                        "loop_times": peak_loop_times,
                        "hit_frames": peak_hit_frames,
                        "gap_times": collision_gap_times,
                        "supplement": "可能有的没敲到方块",
                    }
                else:
                    loop_info = {
                        "loop_times": collision_loop_times,
                        "hit_frames": collision_hit_frames,
                        "gap_times": collision_gap_times,
                        "supplement": "峰值检测可能漏检了一些敲击",
                    }
            else:
                loop_info = {
                    "loop_times": collision_loop_times,
                    "hit_frames": collision_hit_frames,
                    "gap_times": collision_gap_times,
                    "method": "state_machine_contact_detection",
                }
            
            if debug:
                cprint("\n===== Loop Analysis Result =====", "cyan", attrs=["bold"])
                cprint(f"碰撞检测敲击次数: {collision_loop_times}", "yellow")
                cprint(f"峰值检测敲击次数: {peak_loop_times}", "yellow")
                
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
                "hit_frames": [],
                "method": "collision_detection",
                "error_msg": str(e)
            }
        
        # 保存到json
        file_path = f"{self.eval_video_path}/episode{self.test_num}_loop_info.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for key, value in loop_info.items():
                f.write(f"{key}: {value}\n")
        return loop_info



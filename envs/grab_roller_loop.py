from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy


class grab_roller_loop(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.loop_counter = 0
        self.loop_metric = {}

    def load_actors(self):
        ori_qpos = [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0, 0, 0.707, 0.707]]
        self.model_id = np.random.choice([0, 2], 1)[0]
        rand_pos = rand_pose(
            xlim=[-0.15, 0.15],
            ylim=[-0.15, -0.02],
            qpos=ori_qpos[self.model_id],
            rotate_rand=False,
            rotate_lim=[0, 0.8, 0],
        )
        self.roller = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="102_roller",
            convex=True,
            model_id=self.model_id,
        )

        self.add_prohibit_area(self.roller, padding=0.1)

        # Register key objects for 6D pose tracking
        self.set_key_objects({"roller": self.roller})

    def play_once(self, loop_times=8):
        # Initialize arm tags for left and right arms
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")

        # Grasp the roller with both arms simultaneously at different contact points
        self.move(
            self.grasp_actor(self.roller, left_arm_tag, pre_grasp_dis=0.08, contact_point_id=0),
            self.grasp_actor(self.roller, right_arm_tag, pre_grasp_dis=0.08, contact_point_id=1),
        )

        # Align left and right hands to same y coordinate
        left_pose = self.get_arm_pose(left_arm_tag)
        right_pose = self.get_arm_pose(right_arm_tag)
        
        # Move the hand with lower y coordinate to match the higher one
        if left_pose[1] < right_pose[1]:
            # Left hand moves to match right hand's y
            self.move(self.move_by_displacement(left_arm_tag, y=right_pose[1] - left_pose[1]))
        elif right_pose[1] < left_pose[1]:
            # Right hand moves to match left hand's y
            self.move(self.move_by_displacement(right_arm_tag, y=left_pose[1] - right_pose[1]))

        # 先将擀面杖移动到桌子中心位置 (x=0, y=0)
        current_pose = self.roller.get_pose().p
        center_x = 0 - current_pose[0]
        center_y = 0 - current_pose[1]
        self.move(
            self.move_by_displacement(left_arm_tag, x=center_x, y=center_y),
            self.move_by_displacement(right_arm_tag, x=center_x, y=center_y),
        )

        # 再次双手抓紧擀面杖
        # self.move(
        #     self.grasp_actor(self.roller, left_arm_tag, pre_grasp_dis=0.02, contact_point_id=0),
        #     self.grasp_actor(self.roller, right_arm_tag, pre_grasp_dis=0.02, contact_point_id=1),
        # )

        self.wait(0.5)

        # Perform horizontal rolling motion on the table surface (forward and backward)
        for _ in range(loop_times):
            # Move forward horizontally on table surface
            self.move(
                self.move_by_displacement(left_arm_tag, y=0.08),
                self.move_by_displacement(right_arm_tag, y=0.08),
            )
            # Move backward horizontally on table surface
            self.move(
                self.move_by_displacement(left_arm_tag, y=-0.08),
                self.move_by_displacement(right_arm_tag, y=-0.08),
            )

            self.loop_counter += 1

        # Move the roller to the center of the table (x=0, y=0)
        current_pose = self.roller.get_pose().p
        center_x = 0 - current_pose[0]
        center_y = 0 - current_pose[1]
        self.move(
            self.move_by_displacement(left_arm_tag, x=center_x, y=center_y),
            self.move_by_displacement(right_arm_tag, x=center_x, y=center_y),
        )

        # Release sequence: 1) Open grippers, 2) Move hands up, 3) Return to origin
        # Step 1: Open grippers to release the roller
        self.move(
            self.open_gripper(left_arm_tag),
            self.open_gripper(right_arm_tag),
        )
        
        # Step 2: Move hands up to avoid hitting the roller
        self.move(
            self.move_by_displacement(left_arm_tag, z=0.1),
            self.move_by_displacement(right_arm_tag, z=0.1),
        )
        
        # Step 3: Move arms back to origin
        self.move(
            self.back_to_origin(left_arm_tag),
            self.back_to_origin(right_arm_tag),
        )

        # Wait for a short period after completing the rolling motion
        self.wait(0.5)
        
        # Record information about the roller in the info dictionary
        self.info["info"] = {"{A}": f"102_roller/base{self.model_id}"}
        return self.info

    def check_success(self):
        # roller_pose = self.roller.get_pose().p
        # # Check if roller is near the center of table (x≈0, y≈0) and arms are back to origin
        # center_tolerance = 0.1
        # is_at_center = (abs(roller_pose[0]) < center_tolerance and abs(roller_pose[1]) < center_tolerance)
        # arms_released = (not self.is_left_gripper_close() and not self.is_right_gripper_close())
        # return is_at_center and arms_released

        return True

    def record_loop_metric(self):
        if "roller_p" not in self.loop_metric:
            self.loop_metric["roller_p"] = []
            self.loop_metric["arm_pos_left"] = []
            self.loop_metric["arm_pos_right"] = []
            
        roller_p = self.roller.get_pose().p
        arm_pose_left = self.get_arm_pose(ArmTag("left"))
        arm_pose_right = self.get_arm_pose(ArmTag("right"))
        
        self.loop_metric["roller_p"].append(roller_p)
        self.loop_metric["arm_pos_left"].append(arm_pose_left)
        self.loop_metric["arm_pos_right"].append(arm_pose_right)

    def analyze_loop_metric(self):
        try:
            # 保存到文件，方便后续调试
            np.save(f"{self.eval_video_path}/episode{self.test_num}.npz", self.loop_metric)
            
            from .utils.analyze_tools.peak_detect import peak_detect

            if "roller_p" not in self.loop_metric or len(self.loop_metric["roller_p"]) < 10:
                print("[Loop Metric] Not enough data to analyze loop.")
                return {
                    "loop_times": -1,
                    "gap_times": [],
                    "peak_positions": []
                }
                
            # ⭐ 转换为 numpy 数组
            roller_p = np.array(self.loop_metric['roller_p'])
            arm_pos_left = np.array(self.loop_metric['arm_pos_left'])
            arm_pos_right = np.array(self.loop_metric['arm_pos_right'])

            y_distance=roller_p[:, 1]

            # 截断
            # 以roller_p的y轴坐标的绝对值首次小于某个阈值的时间步作为起点
            # 将阈值设为0.05和初始y坐标绝对值的一半的最小值
            init_y_distance = np.abs(y_distance[0])
            threshold = min(0.05, init_y_distance / 2)
            start_index = np.where(np.abs(y_distance) < threshold)[0]
            if start_index.size > 0:
                start_index = start_index[0]
                roller_p = roller_p[start_index:]
                arm_pos_left = arm_pos_left[start_index:]
                arm_pos_right = arm_pos_right[start_index:]
                
            # 将arm_pos_left和arm_pos_right的z轴坐标均大于0.90的时间步作为终点
            end_index = np.where((arm_pos_left[:, 2] > 0.95) & (arm_pos_right[:, 2] > 0.90))[0]
            if end_index.size > 0:
                end_index = end_index[0]
                roller_p = roller_p[:end_index]
                arm_pos_left = arm_pos_left[:end_index]
                arm_pos_right = arm_pos_right[:end_index]

            # 检测y轴距离的波峰
            y_distance = np.array(roller_p[:, 1])
            num_peaks, peak_positions = peak_detect(y_distance, save_plot=True, smooth_window=10, distance_factor=20, save_path=f"{self.eval_video_path}/episode{self.test_num}.png")

            print(f"[Loop Metric] Peak Detection Result: {num_peaks}")
            print(f"[Loop Metric] Peaks at positions: {peak_positions}")

            # ⭐ 将 peak_positions 转换为 Python int 列表，避免 numpy 类型问题
            peak_positions = [int(p) for p in peak_positions]

            loop_info = {
                "loop_times": num_peaks,
                "gap_times": np.diff(peak_positions).tolist() if num_peaks > 1 else [],
                "peak_positions": peak_positions
            }
        except Exception as e:
            print(f"[Loop Metric] Error during analysis: {e}")
            loop_info = {
                "loop_times": -1,
                "gap_times": [],
                "peak_positions": [],
                "error_msg": str(e)
            }
        
        # 先保存，方便调试
        # np.savez(f"/home/haoran/data/code/LoopBreaker/tmp/grl/{time.time()}.npz", **self.loop_metric)
        

        file_path = f"{self.eval_video_path}/episode{self.test_num}_loop_info.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for key, value in loop_info.items():
                f.write(f"{key}: {value}\n")
        return loop_info

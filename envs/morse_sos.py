from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import math
import time

class morse_sos(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.loop_metric={}

    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.15, 0.15],
            ylim=[-0.1, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        while abs(rand_pos.p[0]) < 0.05:
            rand_pos = rand_pose(
                xlim=[-0.15, 0.15],
                ylim=[-0.1, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
            )

        self.bell_id = np.random.choice([0, 1], 1)[0]
        self.bell = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )

        self.add_prohibit_area(self.bell, padding=0.07)
        
        # Initialize SOS morse code tap tracking
        self.tap_count = 0
        self.successful_taps = 0
        self.total_taps = 9  # SOS has 9 total taps
    
    def play_once(self):
        # Choose the arm to use: right arm if the bell is on the right side (positive x), left otherwise
        self.arm_tag = ArmTag("right" if self.bell.get_pose().p[0] > 0 else "left")
    
        # Move the gripper above the top center of the bell
        self.move(self.grasp_actor(
            self.bell,
            arm_tag=self.arm_tag,
            pre_grasp_dis=0.1,
            grasp_dis=0.1,
            contact_point_id=0,  # Targeting the bell's top center
        ))
        
        # SOS Morse code: ... --- ...
        # S = ··· (3 short taps)
        # O = ——— (3 long taps)  
        # S = ··· (3 short taps)
        morse_pattern = [
            'short', 'short', 'short',  # S
            'long', 'long', 'long',     # O
            'short', 'short', 'short'   # S
        ]
        
        self.tap_count = 0
        self.total_taps = len(morse_pattern)
        
        for i, tap_type in enumerate(morse_pattern):
            # Tap the bell
            if tap_type == 'short':
                # Short tap: quick down and up
                self.move(self.move_by_displacement(self.arm_tag, z=-0.045))
                self.check_success()
                self.move(self.move_by_displacement(self.arm_tag, z=0.045))
                # Short pause after short tap
                self.wait(0.05)
            else:  # long tap
                # Long tap: slower down, hold, then up
                self.move(self.move_by_displacement(self.arm_tag, z=-0.045))
                self.check_success()
                # Hold position for longer duration
                self.wait(0.3)
                self.move(self.move_by_displacement(self.arm_tag, z=0.045))
                # Longer pause after long tap
                self.wait(0.05)
            
            self.tap_count += 1
            
            # Add longer pause between letters (S, O, S)
            if i == 2 or i == 5:  # After first S and O
                self.wait(0.3)
            elif i < len(morse_pattern) - 1:  # Between taps within same letter
                self.wait(0.05)

        # Final check for overall success
        self.check_success()

        self.wait(5)  # Wait a moment

        # 保存为npz文件，方便调试
        # np.savez(f"/home/lemonhdl/workspace/RoboTwin/traj/{time.time()}.npz", **self.loop_metric)

        # Record which bell and arm were used in the info dictionary
        self.info["info"] = {"{A}": f"050_bell/base{self.bell_id}", "{a}": str(self.arm_tag)}
        return self.info


    def check_success(self):
        # Initialize tap count and success tracking if not exists
        # if not hasattr(self, 'tap_count'):
        #     self.tap_count = 0
        # if not hasattr(self, 'successful_taps'):
        #     self.successful_taps = 0
        # if not hasattr(self, 'total_taps'):
        #     self.total_taps = 9  # SOS has 9 total taps
            
        # bell_pose = self.bell.get_contact_point(0)[:3]
        # positions = self.get_gripper_actor_contact_position("050_bell")
        # eps = [0.025, 0.025]
        
        # # Check if current tap was successful
        # current_tap_success = False
        # for position in positions:
        #     if (np.all(np.abs(position[:2] - bell_pose[:2]) < eps) and abs(position[2] - bell_pose[2]) < 0.03):
        #         current_tap_success = True
        #         break
        
        # if current_tap_success and not hasattr(self, f'tap_{self.tap_count}_success'):
        #     # Mark this specific tap as successful (avoid double counting)
        #     setattr(self, f'tap_{self.tap_count}_success', True)
        #     self.successful_taps += 1
        
        # # Task is successful if we completed all 9 taps with at least 7 successful
        # if hasattr(self, 'tap_count') and self.tap_count >= self.total_taps:
        #     self.stage_success_tag = (self.successful_taps >= 7)  # Allow some tolerance
        #     return self.stage_success_tag
        
        # return current_tap_success

        return True
    
    # -------- loop -------- 
    def record_loop_metric(self):
        # 拿到arm_tag对应的手臂末端数据
        arm_pose=self.get_arm_pose(arm_tag = self.arm_tag)
        # 拿到铃铛的位置
        bell_pose=self.bell.get_pose().p
        
        # 当前手臂进入铃铛上方一定范围才开始记录
        
        if "arm_pose" not in self.loop_metric:
            self.loop_metric["arm_pose"] = []
            self.loop_metric["bell_pose"] = []
        self.loop_metric["arm_pose"].append(arm_pose)
        self.loop_metric["bell_pose"].append(bell_pose)

    def analyze_loop_metric(self):
        # 保存为npz文件，方便调试
        # np.savez(f"/home/lemonhdl/workspace/RoboTwin/eval_traj/{time.time()}.npz", **self.loop_metric)
        try:
            # 保存到文件，方便后续调试
            np.save(f"{self.eval_video_path}/episode{self.test_num}.npz", self.loop_metric)

            
            # pass
            from .utils.analyze_tools.peak_detect import peak_detect       
            from .utils.analyze_tools.wavelength_detect import wavelength_detect 

            # 取出记录的数据
            arm_poses = np.array(self.loop_metric["arm_pose"])  # shape: (N, 7)
            bell_poses = np.array(self.loop_metric["bell_pose"])  # shape: (N, 3)
            
            # 数据预处理 先转换为numpy array
            arm_poses = np.array(arm_poses)  # shape: (N, 7)
            bell_poses = np.array(bell_poses)  # shape: (N, 3)
            
            # 计算手臂末端与铃铛位置的高度差
            height_z = arm_poses[:, 2] - bell_poses[:, 2]
            
            ##################### 截断前后多余部分 #####################
            # 当夹爪x,y位置第一次在铃铛x,y位置的正上方一定范围内
            start_index=np.where(
                np.sqrt(np.abs(arm_poses[:,0]-bell_poses[:,0])**2+np.abs(arm_poses[:,1]-bell_poses[:,1])**2)<0.05
            )[0][0]
            
            arm_poses=arm_poses[start_index:]
            bell_poses=bell_poses[start_index:]

            # 当夹爪x,y位置第一次离开铃铛x,y位置的正上方一定范围内
            end_index=np.where(
                np.sqrt(np.abs(arm_poses[1:,0]-bell_poses[1:,0])**2+np.abs(arm_poses[1:,1]-bell_poses[1:,1])**2)>0.055
            )[0][0]
            arm_poses=arm_poses[:end_index]
            bell_poses=bell_poses[:end_index]

            arm_poses = - arm_poses
            height_z = arm_poses[:, 2] - bell_poses[:, 2]
            num_peaks, peak_positions = peak_detect(
                height_z, 
                smooth=True, 
                smooth_window=15,
                height_factor=0.2, 
                distance_factor=20, 
                prominence_factor=0.02, 
                save_plot=True,
                save_path=f"{self.eval_video_path}/episode{self.test_num}.png"
            )

            # 只有在num_peaks == 9时才进行波长检测
            wave_info={}
            if num_peaks == 9:
                wave_info = wavelength_detect(
                        height_z, 
                        save_plot=True,
                        save_path=f"{self.eval_video_path}/episode{self.test_num}_wavelength.png")
                    
                    # 提取并处理wave_info中的波长数据
                wavelengths = [wave['wavelength'] for wave in wave_info]
                wave_num = len(wavelengths)
                # 将最大的3个波长和最小的3个波长的索引保存为字典
                top_3_wave_indices = np.argsort(wavelengths)[-3:][::-1].tolist() if wave_num >= 3 else np.argsort(wavelengths).tolist()
                bottom_3_wave_indices = np.argsort(wavelengths)[:3].tolist() if wave_num >= 3 else np.argsort(wavelengths).tolist()
                top_3_wave_lengths = [wavelengths[i] for i in top_3_wave_indices]
                bottom_3_wave_lengths = [wavelengths[i] for i in bottom_3_wave_indices]
                top_3_waves= {
                    "order": top_3_wave_indices,
                    "length": top_3_wave_lengths
                }
                lower_3_waves= {
                    "order": bottom_3_wave_indices,
                    "length": bottom_3_wave_lengths
                }   
                middle_3_waves= {
                    "order": np.argsort(wavelengths)[wave_num//2 - 1: wave_num//2 + 2].tolist() if wave_num >=3 else np.argsort(wavelengths).tolist(),
                    "length": [wavelengths[i] for i in (np.argsort(wavelengths)[wave_num//2 - 1: wave_num//2 + 2].tolist() if wave_num >=3 else np.argsort(wavelengths).tolist())]
                }
                
                wave_info = {
                    "wavelengths": wavelengths,
                    "wave_num": wave_num,
                    "top_3_waves": top_3_waves,
                    "lower_3_waves": lower_3_waves,
                    "middle_3_waves": middle_3_waves
                }
            
            
            print(f"[Loop Metric] Peak Detection Result: {num_peaks}")
            print(f"[Loop Metric] Peaks at positions: {peak_positions}")

            loop_info = {
                "loop_times": num_peaks,
                "gap_times": np.diff(peak_positions).tolist() if num_peaks > 1 else [],
                "peak_positions": peak_positions,
                "wave_info": wave_info
            }
        except Exception as e:
            print(f"[Loop Metric] Error in analyze_loop_metric: {e}")
            loop_info = {
                "loop_times": -1,
                "gap_times": [],
                "peak_positions": []
            }
        return loop_info
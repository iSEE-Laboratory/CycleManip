from ._base_task import Base_Task
from .utils import *
import sapien
from copy import deepcopy


class beat_egg_loop(Base_Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = 0
        self.loop_counter = 0

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.bowl_id=np.random.choice([0,1,2,3,4,5],1)[0]
        bowl_pose_p=[
            np.random.uniform(-0.2, -0.1),
            np.random.uniform(-0.15, 0),
            0.742 
        ]
        bowl_pose_q=[0.631995, 0.631983, 0.317149, 0.31717] # magic number :(

        #打蛋器相对碗的初始位置一定
        self.egg_beater_pose_p=[bowl_pose_p[0] + 0.2056206, bowl_pose_p[1] - 0.081223, 1.26503]
        self.egg_beater_pose_q = [0, 1, 0, 0]

        self.bowl = create_actor(
            scene=self.scene,
            pose=sapien.Pose(bowl_pose_p, bowl_pose_q),
            modelname="136_egg_bowl", #实际上是马克杯 :)
            model_id=self.bowl_id,
            convex=True,
        )
        self.bowl.set_mass(0.002)
        
        rand_pos = sapien.Pose(
            p=[self.egg_beater_pose_p[0], self.egg_beater_pose_p[1], 0.739933],
            q=[0.710204, 0.703993, 0.00166242, -0.0011352]
        )
        self.cup = create_actor(
            scene=self.scene,
            pose=rand_pos,
            modelname="021_cup", 
            model_id=0,
            convex=True,
            is_static=True,
        )

        self.delay(3)
        self.egg_beater = create_actor(
            scene=self.scene,
            pose=sapien.Pose(self.egg_beater_pose_p, self.egg_beater_pose_q),
            modelname="137_egg_beater",
            convex=True,
        )
        self.egg_beater.set_mass(0.005) 
        bowl_pose = self.bowl.get_pose()
        
        # 生成黄球和白球（1：2）
        self.egg_num = 112
        self.sphere_lst = []
        for i in range(self.egg_num):
            if i%30==0:
                self.delay(5)
            if i%3==0:
                sphere_pose = sapien.Pose(
                    [
                        bowl_pose.p[0] + np.random.rand() * 0.03 - 0.015,
                        bowl_pose.p[1] + np.random.rand() * 0.02 - 0.02,
                        0.742 + self.table_z_bias + i * 0.005,
                    ],
                    [1, 0, 0, 0],
                )
            else:
                sphere_pose = sapien.Pose(
                    [
                        bowl_pose.p[0] + np.random.rand() * 0.03 + 0.005,
                        bowl_pose.p[1] + np.random.rand() * 0.02,
                        0.742 + self.table_z_bias + i * 0.005,
                    ],
                    [1, 0, 0, 0],
                )
            color = [1, 1, 0] if i%3 == 0 else [1, 1, 1]
            sphere = create_sphere(
                self.scene,
                pose=sphere_pose,
                radius=0.0063,
                color=color,
                name="eggs",
            )
            self.sphere_lst.append(sphere)
            self.sphere_lst[-1].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.000045

        self.set_key_objects({"egg_beater": self.egg_beater, "bowl": self.bowl})

    def play_once(self, loop_times=3):
        print("bowl id: ", self.bowl_id)
        self.egg_beater_pose_p=self.egg_beater.get_pose().p
        
        grasp_egg_beater_arm_tag = ArmTag("right")
        hold_bowl_arm_tag = ArmTag("left")
        
        # 抓取碗后抬起
        self.move(
            self.grasp_actor(
                self.bowl,
                arm_tag=hold_bowl_arm_tag,
                pre_grasp_dis=0.08,
        ))
        
        self.move(self.move_by_displacement(arm_tag=hold_bowl_arm_tag, z=0.03, move_axis="world"))
        self.delay(10)
        
        
        # 抓取打蛋器后移开
        self.move(
            self.grasp_actor(
                self.egg_beater,
                arm_tag=grasp_egg_beater_arm_tag,
                pre_grasp_dis=0.08,
            ))
        self.delay(1)
        self.move(self.move_by_displacement(arm_tag=grasp_egg_beater_arm_tag, z=0.24, move_axis="world"))
        self.move(self.move_by_displacement(arm_tag=grasp_egg_beater_arm_tag, x=0.12, move_axis="world"))
        self.move(self.move_by_displacement(arm_tag=grasp_egg_beater_arm_tag, y=-0.1, move_axis="world"))
        self.delay(5)
        
        print(self.egg_beater.get_pose())
        print(self.bowl.get_pose())
        
        bowl_pose = self.bowl.get_pose().p
        target_pose = bowl_pose + np.array([0.08, 0.02, -0.02])
 
        # 将碗移动到合适的位置
        self.move(
            self.place_actor(
                self.bowl,
                target_pose=target_pose.tolist()+[0.7071, 0, 0, 0.7071],
                arm_tag=hold_bowl_arm_tag,
                pre_dis=0.1,
                dis=0,
                constrain="align",
                is_open=False,
            ))
        print(self.bowl.get_pose())

        bowl_pose = self.bowl.get_pose().p

        # 将打蛋器移动至碗的上方
        target_pose = bowl_pose + np.array([0.015, 0.05, 0.048])
        self.move(
            self.place_actor(
                self.egg_beater,
                target_pose=target_pose.tolist()+[0.9659258262890683, 0.25881904510252074, 0, 0],
                arm_tag=grasp_egg_beater_arm_tag,
                pre_dis=0.05,
                dis=0.02,
                constrain="align",
                is_open=False,
            ))
        self.move(self.move_by_displacement(arm_tag=grasp_egg_beater_arm_tag, z=-0.024, move_axis="world"))
    
        for i in range(loop_times):
            # 执行一次搅拌动作
            self.move(self.move_by_displacement(arm_tag=grasp_egg_beater_arm_tag, x=0.044, move_axis="world"))
            self.move(self.move_by_displacement(arm_tag=grasp_egg_beater_arm_tag, y=0.042, move_axis="world"))
            self.move(self.move_by_displacement(arm_tag=grasp_egg_beater_arm_tag, x=-0.044, move_axis="world"))
            if i<4:
                self.move(self.move_by_displacement(arm_tag=grasp_egg_beater_arm_tag, z=-0.001, move_axis="world"))
            self.move(self.move_by_displacement(arm_tag=grasp_egg_beater_arm_tag, y=-0.042, move_axis="world"))
            self.loop_counter = self.loop_counter + 1
            self.delay(10)
        
        # 搅拌完成后抬起打蛋器
        self.delay(10)
        self.move(self.move_by_displacement(arm_tag=grasp_egg_beater_arm_tag, z=0.1, move_axis="world"))
        self.move(self.move_by_displacement(arm_tag=grasp_egg_beater_arm_tag, z=0.05, y=-0.1, x=0.2, move_axis="world"))
        print(self.egg_beater.get_pose())
        self.delay(3)
        self.info["info"] = {"loop_times": f"{loop_times}"}
        return self.info

    def check_success(self):
        return self.egg_beater.get_pose().p[2]>0.9

    def record_loop_metric(self):
        # 拿到arm_tag对应的手臂末端数据
        bowl_pose=self.bowl.get_pose().p
        egg_beater_pose=self.egg_beater.get_pose().p
        left_arm_pose=self.get_arm_pose(ArmTag("left"))  # 碗手臂
        right_arm_pose=self.get_arm_pose(ArmTag("right"))  # 打蛋器手臂

        if "bowl_pose" not in self.loop_metric:
            self.loop_metric["bowl_pose"] = []
            self.loop_metric["egg_beater_pose"] = []
            self.loop_metric["left_arm_pose"] = []
            self.loop_metric["right_arm_pose"] = []
        self.loop_metric["bowl_pose"].append(bowl_pose)
        self.loop_metric["egg_beater_pose"].append(egg_beater_pose)
        self.loop_metric["left_arm_pose"].append(left_arm_pose)
        self.loop_metric["right_arm_pose"].append(right_arm_pose)

    def analyze_loop_metric(self):
        # 将 loop_metric 保存为.npz文件
        # 先创建文件夹
        import time
        import os
        # os.makedirs("//home/liaohaoran/code/LoopBreaker/tmp/bel/1762673286.1726553.npz", exist_ok=True)
        # np.savez(f"/home/liaohaoran/code/LoopBreaker/tmp/bel/{time.time()}.npz", **self.loop_metric)
        
        try:
            # 保存到文件，方便后续调试
            np.save(f"{self.eval_video_path}/episode{self.test_num}.npz", self.loop_metric)
            # 取出记录的数据
            bowl_poses = np.array(self.loop_metric["bowl_pose"])  # shape: (N, 7)
            egg_beater_poses = np.array(self.loop_metric["egg_beater_pose"])  # shape: (N, 7)
            right_arm_poses = np.array(self.loop_metric["right_arm_pose"])  # shape: (N, 7)
            
            # 截断前后多余部分

            start_index = np.where(egg_beater_poses[:,0]>0.1)[0][0]
            bowl_poses=bowl_poses[start_index:]
            egg_beater_poses=egg_beater_poses[start_index:]
            right_arm_poses=right_arm_poses[start_index:]

            start_index = np.where(right_arm_poses[:,2]<0.95)[0][0]
            bowl_poses=bowl_poses[start_index:]
            egg_beater_poses=egg_beater_poses[start_index:]
            right_arm_poses=right_arm_poses[start_index:]
            
            end_index= np.where(right_arm_poses[:,2]>0.95)[0]
            if len(end_index)>0:
                end_index = end_index[0]
            else:
                end_index = len(egg_beater_poses)

            bowl_poses=bowl_poses[:end_index]
            egg_beater_poses=egg_beater_poses[:end_index]
            right_arm_poses=right_arm_poses[:end_index]

            # 检测 x + y 的峰值
            beater_pos_metrics = egg_beater_poses[:, 0] + egg_beater_poses[:, 1]
            
            from .utils.analyze_tools.peak_detect import peak_detect
            
            num_peaks, peak_positions = peak_detect(
                beater_pos_metrics,
                smooth=True,
                smooth_window=30,
                height_factor=0.1, 
                distance_factor=20, 
                prominence_factor=0.02, 
                save_plot=False,
            )
            
            # 利用检测到的峰值计算 拟合一条直线
            
            if num_peaks > 3:
                peak_heights = beater_pos_metrics[peak_positions]
                coefficients = np.polyfit(peak_positions, peak_heights, 1)
                slope, intercept = coefficients[0], coefficients[1]
            else:
                slope = None
                intercept = None
                
            # 将metrics减去拟合的直线部分
            if slope is not None:
                fitted_line = slope * np.array(range(len(beater_pos_metrics))) + intercept
                adjusted_metrics = beater_pos_metrics - fitted_line
            else:
                adjusted_metrics = beater_pos_metrics
                
            # 再次检测峰值
            num_peaks, peak_positions = peak_detect(
                adjusted_metrics,
                smooth=True,
                smooth_window=30,
                height_factor=0.05, 
                distance_factor=10, 
                prominence_factor=0.01, 
                save_plot=True,
                save_path=f"{self.eval_video_path}/episode{self.test_num}.png"
            )
            
            loop_info = {
                "loop_times": num_peaks,
                "gap_times": np.diff(peak_positions).tolist() if num_peaks > 1 else [],
                "peak_positions": peak_positions
            }
        except Exception as e:
            print(f"[Loop Metric] Error in analyze_loop_metric: {e}")
            loop_info = {
                "loop_times": -1,
                "gap_times": [],
                "peak_positions": []
            }
        file_path = f"{self.eval_video_path}/episode{self.test_num}_loop_info.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for key, value in loop_info.items():
                f.write(f"{key}: {value}\n")
        return loop_info
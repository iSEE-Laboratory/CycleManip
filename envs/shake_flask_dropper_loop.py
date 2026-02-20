from ._base_task import Base_Task
from .utils import *
import sapien
import math
import time
from scipy.spatial.transform import Rotation as R
            
class shake_flask_dropper_loop(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)
        self.bad=0
        self.kk=0
        
        self.loop_counter = 0
    
    def load_actors(self):
        self.rand_pos = sapien.Pose(
            p=[-0.02523111+np.random.uniform(-0.02, 0.04), -0.17531879+np.random.uniform(-0.02, 0.04), 0.7398382],
            q=[0.703342, 0.71058, -0.0113243, 0.0160776],  # 90-degree rotation around x-axis
        )
        
        # Use the random position from rand_pos, but set rotation to (0,0,0)
        self.flask = create_actor(
            scene=self,
            pose=self.rand_pos,
            modelname="122_flask",
            convex=True,
            scale=0.1,
        )
        self.flask.set_mass(0.077)
        
        rand_pos = sapien.Pose(
            p=[0.155573+np.random.uniform(-0.02, 0.03), -0.0110224+np.random.uniform(-0.03, 0.02), 0.739933],
            q=[0.710204, 0.703993, 0.00166242, -0.0011352]
        )
        self.beaker = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="138_dropper_cup",
            model_id=3,
            convex=True,
            is_static=True,
            scale=0.5,
        )
        self.beaker.set_mass(0.5)
        self.delay(3)
        beaker_pose=self.beaker.get_pose()
        dropper_pose = sapien.Pose(
            p=[beaker_pose.p[0], beaker_pose.p[1], 0.75808],
            q=[0.70710678, 0.70710678, 0.0, 0.0]  # 90° around X: [cos(π/4), sin(π/4)*1, 0, 0]
        )
        self.dropper = create_actor(
            scene=self,
            pose=dropper_pose,
                modelname="123_dropper",
                convex=True,
                scale=0.2,
            )
        self.dropper.set_mass(0.005)

        self.set_key_objects({"dropper": self.dropper, "flask": self.flask})

    def play_once(self, loop_times=3):
        print(self.dropper.get_pose())
        self.flask_pose = self.flask.get_pose()
        arm_tag = ArmTag("left")
        grasp_dropper_arm_tag = ArmTag("right")
        # 拿起滴管并抬起
        self.move(self.grasp_actor(self.dropper, arm_tag=grasp_dropper_arm_tag, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag=grasp_dropper_arm_tag, z=0.25))
        
        self.move(self.grasp_actor(self.flask, arm_tag=arm_tag, pre_grasp_dis=0.12))
        # 设置摇烧瓶前后的目标姿态
        target_quat = np.array(self.get_arm_pose(ArmTag("left"))[3:7])
        
        theta = np.pi / 12
        axis = np.array([1, 0, 1]) / np.sqrt(2)
        w = np.cos(theta / 2)
        x = axis[0] * np.sin(theta / 2)
        y = axis[1] * np.sin(theta / 2)
        z = axis[2] * np.sin(theta / 2)
        r1 = [w, x, y, z]

        theta = -np.pi / 12
        axis = np.array([1, 0, 1]) / np.sqrt(2)
        w = np.cos(theta / 2)
        x = axis[0] * np.sin(theta / 2)
        y = axis[1] * np.sin(theta / 2)
        z = axis[2] * np.sin(theta / 2)
        r2 = [w, x, y, z]

        quat1=t3d.quaternions.qmult(target_quat, r1)
        quat2=t3d.quaternions.qmult(target_quat, r2)

        curr_pos = np.array(self.get_arm_pose(ArmTag("left"))[:3])
        target_quat = np.array(self.get_arm_pose(ArmTag("left"))[3:7])
        
       
        target_pos_d = sapien.Pose(
            p=self.flask.get_pose().p + np.array([-0.01, 0.05, 0.218]),
            q=[0.63613, 0.36707, 0.26423, 0.62751]
        )
        # 将滴管置于烧瓶正上方
        self.move(self.place_actor(self.dropper, arm_tag=grasp_dropper_arm_tag, target_pose=target_pos_d, is_open=False))
        if self.dropper.get_pose().p[2] > 1:
            self.bad=1
        self.delay(18)
        # 作滴液体动作
        for k in range(3):
            self.move(self.move_by_displacement(arm_tag=grasp_dropper_arm_tag, z=0.01))
            self.move(self.move_by_displacement(arm_tag=grasp_dropper_arm_tag, z=-0.01))
            self.delay(6)
        
        # 将滴管移开
        self.move(self.move_by_displacement(arm_tag=grasp_dropper_arm_tag, x=0.2))
        
        # Grasp the flask with specified pre-grasp distance
        # Perform shaking motion three times (alternating between two orientations)
        self.move(self.move_by_displacement(arm_tag=arm_tag, quat=target_quat))
        flask_pose=self.flask.get_pose()
        for i in range(loop_times):
            if self.bad==1:
                break
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05, quat=quat1))
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.05, quat=quat2))
            self.loop_counter += 1
            if self.loop_counter==loop_times:
                self.kk=1
                break

            #if self.kk == 0:
            #    self.make_flask_stable(flask_pose)
            
        if self.kk==1:
            # 将烧瓶移回原位
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.02, quat=target_quat))
            target_pos = curr_pos
            error_pos = target_pos - curr_pos
            curr_pos = np.array(self.get_arm_pose(ArmTag("left"))[:3])
            self.move(self.move_by_displacement(arm_tag, x=error_pos[0]+0.08, y=error_pos[1], z=error_pos[2], move_axis="world"))
            self.move(self.move_by_displacement(arm_tag, z=-0.01, move_axis="world"))
            self.delay(10)
            self.move(self.open_gripper(arm_tag))
            self.delay(10)
            self.move(self.move_by_displacement(arm_tag, x=-0.04,y=-0.04, move_axis="world"))
            self.delay(10)
            self.move(self.move_by_displacement(arm_tag, x=-0.008,y=-0.008, z=0.2, move_axis="world"))
            self.delay(20)
            self.move(self.move_by_displacement(arm_tag, y=-0.05, x=-0.1, move_axis="world"))
            self.delay(10)
            
        self.info["info"] = {
            "{A}": "122_flask/base0"
        }
        return self.info

    def check_success(self):
        return (
            self.bad==0
            and abs(self.flask.get_pose().p[2] - self.flask_pose.p[2]) < 0.01
            and self.dropper.get_pose().p[2] > 0.9
        )
        
    def record_loop_metric(self):
        # 拿到烧瓶的位置和朝向
        flask_p = self.flask.get_pose().p
        flask_q = self.flask.get_pose().q
        dropper_p = self.dropper.get_pose().p
        dropper_q = self.dropper.get_pose().q
        left_arm_pose = self.get_arm_pose(ArmTag("left"))
        right_arm_pose = self.get_arm_pose(ArmTag("right"))
    
        # 如果没有记录过，初始化
        if "flask_p" not in self.loop_metric:
            self.loop_metric["flask_p"] = []
            self.loop_metric["flask_q"] = []
            self.loop_metric["dropper_p"] = []
            self.loop_metric["dropper_q"] = []
            self.loop_metric["left_arm_pose"] = []
            self.loop_metric["right_arm_pose"] = []

        self.loop_metric["flask_p"].append(flask_p.copy())
        self.loop_metric["flask_q"].append(flask_q.copy())
        self.loop_metric["dropper_p"].append(dropper_p.copy())
        self.loop_metric["dropper_q"].append(dropper_q.copy())
        self.loop_metric["left_arm_pose"].append(left_arm_pose.copy())
        self.loop_metric["right_arm_pose"].append(right_arm_pose.copy())

    def analyze_loop_metric(self):
        try:
            # 保存到文件，方便后续调试
            np.save(f"{self.eval_video_path}/episode{self.test_num}.npz", self.loop_metric)

            # 取出记录的数据
            flask_p=np.array(self.loop_metric["flask_p"])
            flask_q=np.array(self.loop_metric["flask_q"])
            dropper_p=np.array(self.loop_metric["dropper_p"])
            dropper_q=np.array(self.loop_metric["dropper_q"])
            left_arm_pose=np.array(self.loop_metric["left_arm_pose"])

            # 截断前后多余部分
            start_index = 0
            start_index_1 = np.where((flask_p[:,0]-left_arm_pose[:,0])<0.15)[0][0]
            start_index_2 = np.where(dropper_p[:,0]<0.1)[0][0]

            start_index = max(start_index_1, start_index_2)
            flask_p=flask_p[start_index:]
            dropper_p=dropper_p[start_index:]
            left_arm_pose=left_arm_pose[start_index:]

            start_index = np.where((dropper_p[:,0])>0.1)[0][0]+1
            flask_p=flask_p[start_index:]
            dropper_p=dropper_p[start_index:]
            left_arm_pose=left_arm_pose[start_index:]

            end_index_2 = np.where(left_arm_pose[:,2]>0.97)[0]
            if len(end_index_2)>0:
                end_index_2 = end_index_2[0]
            else:
                end_index_2 = len(left_arm_pose)

            end_index = end_index_2
            end_index = max(min(50,end_index), end_index - 50)

            flask_p=flask_p[:end_index]
            flask_q=flask_q[:end_index]
            dropper_p=dropper_p[:end_index]
            dropper_q=dropper_q[:end_index]
            left_arm_pose=left_arm_pose[:end_index]

            # 对烧瓶x-arm_x进行峰值检测
            flask_metric=(flask_p[:,1]-flask_p[:,0])+left_arm_pose[:,0]-left_arm_pose[:,1]
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
            flask_metric = pid_inspired_filter(flask_metric, kp=0.7, kd=0.25, ki=0.05)
            
            print("flask_metric.shape: ", flask_metric.shape)
            
            from .utils.analyze_tools.peak_detect import peak_detect
            num_peaks, peak_positions = peak_detect(
                flask_metric, 
                smooth=True,    
                smooth_window=21,
                height_factor=0.66, 
                distance_factor=20, 
                prominence_factor=0.12, 
                save_plot=True,
                save_path=f"{self.eval_video_path}/episode{self.test_num}.png"
            )
            
            loop_info = {
                "loop_times": num_peaks,
                "gap_times": np.diff(peak_positions).tolist() if num_peaks > 1 else [],
                "peak_positions": peak_positions
            }

        except Exception as e:
            
            print(f"[Loop Analysis] 分析过程中出现错误: {e}")
            loop_info = {
                "loop_times": -1,
                "gap_times": [],
                "peak_positions": [],
                "error_msg": str(e)
            }

        file_path = f"{self.eval_video_path}/episode{self.test_num}_loop_info.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for key, value in loop_info.items():
                f.write(f"{key}: {value}\n")
        return loop_info

from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


class shake_bottle_loop(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        # 调整相机视角 - 仅对此任务生效
        # 修改 head_camera 的 forward 向量，增加仰角
        if "left_embodiment_config" in kwags:
            for camera_info in kwags["left_embodiment_config"].get("static_camera_list", []):
                if camera_info.get("name") == "head_camera":
                    camera_info["position"] = [-0.032, -0.55, 1.30]
                    camera_info["forward"] = [0, 0.6, -0.4]

                    print(f"[Camera Adjustment] Left head_camera position set to {camera_info['position']}")
                    print(f"[Camera Adjustment] Left head_camera forward set to {camera_info['forward']}")

                    break
        
        super()._init_task_env_(**kwags)
        self.end = False

        self.loop_counter = 0

    def load_actors(self):
        self.id_list = [i for i in range(20)]
        rand_pos = rand_pose(
            xlim=[-0.15, 0.15],
            ylim=[-0.1, -0.05],
            zlim=[0.785],
            qpos=[0, 0, 1, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 4],
        )
        while abs(rand_pos.p[0]) < 0.1:
            rand_pos = rand_pose(
                xlim=[-0.15, 0.15],
                ylim=[-0.1, -0.05],
                zlim=[0.785],
                qpos=[0, 0, 1, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, np.pi / 4],
            )
        self.bottle_id = np.random.choice(self.id_list)
        self.bottle = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="001_bottle",
            convex=True,
            model_id=self.bottle_id,
        )
        self.bottle.set_mass(0.01)
        self.add_prohibit_area(self.bottle, padding=0.05)

        # Register key objects for 6D pose tracking
        self.set_key_objects({"bottle": self.bottle})

    def play_once(self, loop_times=3):
        # Determine which arm to use based on bottle position
        arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] > 0 else "left")

        # Grasp the bottle with specified pre-grasp distance
        self.move(self.grasp_actor(self.bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))

        # Lift the bottle up by 0.2m while rotating to target orientation
        target_quat = [0.707, 0, 0, 0.707]
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, quat=target_quat))

        # Prepare two shaking orientations by rotating around y-axis
        quat1 = deepcopy(target_quat)
        quat2 = deepcopy(target_quat)
        # First shake rotation (7π/8 around y-axis)
        y_rotation = t3d.euler.euler2quat(0, (np.pi / 8) * 7, 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, quat1)
        quat1 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        # Second shake rotation (-7π/8 around y-axis)
        y_rotation = t3d.euler.euler2quat(0, -7 * (np.pi / 8), 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, quat2)
        quat2 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        # Perform shaking motion three times (alternating between two orientations)
        for _ in range(loop_times):
            # Move up with first shaking orientation
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05, quat=quat1))
            # Move down with second shaking orientation
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.05, quat=quat2))

            self.loop_counter += 1

        # Return to original grasp orientation
        self.move(self.move_by_displacement(arm_tag=arm_tag, quat=target_quat))
        
        # return to the center of the table
        self.move(self.back_to_origin(arm_tag=arm_tag))
        
        # 在初始位置停留一小段时间
        self.wait(0.5)

        print(self.bottle.get_pose().p, self.bottle.get_pose().q)
        
        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        bottle_pose = self.bottle.get_pose().p
        return bottle_pose[2] > 0.8 + self.table_z_bias


    # -------- loop -------- 
    def record_loop_metric(self):
        def get_object_normal(quaternion):
            """
            获取物体的法方向（z轴方向）
            
            Args:
                position: [x, y, z] 位置
                quaternion: [qx, qy, qz, qw] 四元数
            
            Returns:
                normal_vector: 法方向向量 [nx, ny, nz]
            """
            # 创建旋转对象
            rotation = R.from_quat(quaternion)
            
            # 获取旋转矩阵
            rotation_matrix = rotation.as_matrix()
            
            # 物体的z轴在局部坐标系中是 [0, 0, 1]
            # 转换到世界坐标系
            local_z = np.array([0, 0, 1])
            normal_vector = rotation_matrix @ local_z
            
            return normal_vector
        
        # 拿到瓶子的位置和朝向
        bottle_p = self.bottle.get_pose().p
        bottle_q = self.bottle.get_pose().q
        arm_left = self.get_arm_pose(ArmTag("left"))
        arm_right = self.get_arm_pose(ArmTag("right"))

        if "bottle_p" not in self.loop_metric:
            self.loop_metric["bottle_p"] = []
            self.loop_metric["bottle_q"] = []
            self.loop_metric["arm_left"] = []
            self.loop_metric["arm_right"] = []
        else:
            self.loop_metric["bottle_p"].append(bottle_p)
            self.loop_metric["bottle_q"].append(bottle_q)
            self.loop_metric["arm_left"].append(arm_left)
            self.loop_metric["arm_right"].append(arm_right)

        if bottle_p[1] < -0.1: # 如果瓶子y轴位置太小，说明已经拿回来了，结束
            self.end = True

        if bottle_p[2] < 0.785 or self.end: # 如果瓶子高度和设置的一样，说明还没拿起来，不记录
            return

        # 获取其法方向
        bottle_normal_vector = get_object_normal(bottle_q)
        # print(f"[Loop Metric] Bottle Z-Axis: {bottle_p}, {bottle_q}")

        # 投影到xy平面上(其实就是去看z轴和z轴的夹角)
        bottle_normal_vector[2] = 0
        bottle_normal_vector = bottle_normal_vector / np.linalg.norm(bottle_normal_vector)
        # print(f"[Loop Metric] Bottle Z-Axis Normal: {bottle_normal_vector}")
        
        # 如果没有记录过，初始化
        if "normal_y" not in self.loop_metric:
            self.loop_metric["normal_y"] = []

        # 通过观察图像，y轴的变化更明显
        self.loop_metric["normal_y"].append(bottle_normal_vector[1].copy())


        # 保存其他参数

    def analyze_loop_metric(self):

        try:
            # pass
            from .utils.analyze_tools.peak_detect import peak_detect

            # 保存到文件，方便后续调试
            np.save(f"{self.eval_video_path}/episode{self.test_num}.npz", self.loop_metric)

            if "normal_y" not in self.loop_metric:
                print(f"[Loop Metric] No normal_y data recorded.")
                return {
                    "loop_times": 0,
                    "gap_times": [],
                    "peak_positions": []
                }
            if len(self.loop_metric["normal_y"]) < 10:
                print(f"[Loop Metric] Not enough normal_y data recorded: {len(self.loop_metric['normal_y'])} samples.")
                return {
                    "loop_times": 0,
                    "gap_times": [],
                    "peak_positions": []
                }
            
            normal_y = np.array(self.loop_metric["normal_y"])
            # print(normal_y.shape, normal_y)


            # 保存为npy文件，方便调试
            # np.save(f"/home/haoran/tmp/traj/{time.time()}.npy", normal_y)

            normal_y = np.array(normal_y)
            num_peaks, peak_positions = peak_detect(
                normal_y, 
                smooth=True, 
                smooth_window=20,
                height_factor=0.4, 
                distance_factor=30, 
                prominence_factor=0.075, 
                save_plot=True,
                save_path=f"{self.eval_video_path}/episode{self.test_num}.png"
            )

            # print(f"[Loop Metric] Peak Detection Result: {num_peaks}")
            # print(f"[Loop Metric] Peaks at positions: {peak_positions}")

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


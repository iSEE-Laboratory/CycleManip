import cv2
import os
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

def adaptive_fixthre_endpose_sample_indices(
        start_idx: int,
        end_idx: int,
        traj: np.ndarray,
        rot_thresh: float = 10.0,       # deg
        trans_thresh: float = 0.015,    # m
        gripper_thresh: float = 0.5,
    ) -> np.ndarray:
    """
    按时间步遍历：对于每个时间 i，检查左右两侧（0/1）相对于它们各自上次被采样时刻
    的累计变化是否超过阈值。如果任一侧超过阈值，则采样 i（global idx = start_idx + i）。
    仅更新触发采样的侧的 last_sample。

    traj: shape (T, 14)  按顺序是 [L_xyz, L_rpy, L_gripper, R_xyz, R_rpy, R_gripper]
    返回全局索引 numpy array (sorted, unique)
    """
    assert 0 <= start_idx < end_idx <= traj.shape[0], f"invalid index, {start_idx}, {end_idx}"

    # 局部索引范围 [0, N)
    N = end_idx - start_idx
    if N <= 0:
        return np.array([], dtype=np.int64)

    # 以局部索引访问切片
    traj_slice = traj[start_idx:end_idx]

    # 初始化 last_sample（局部索引）
    last_sample = [0, 0]  # left, right
    key_indices = [0]     # 保留第一帧（局部索引0）

    # 便利的访问器：给 side 返回对应切片区间 offset
    def get_slice(i, side):
        off = 7 * side
        pos = traj_slice[i, off: off + 3]
        rpy = traj_slice[i, off + 3: off + 6]
        grip = float(traj_slice[i, off + 6])
        return pos, rpy, grip

    # 主循环：按时间步检查两侧
    for i in range(1, N):
        triggered = [False, False]

        for side in (0, 1):
            pos_i, rpy_i, grip_i = get_slice(i, side)
            pos_last, rpy_last, grip_last = get_slice(last_sample[side], side)

            # 平移差
            d_trans = np.linalg.norm(pos_i - pos_last)

            # 旋转差（scipy Rotation 单个对单个）
            r1 = R.from_euler('xyz', rpy_last)
            r2 = R.from_euler('xyz', rpy_i)
            d_rot = np.degrees((r2 * r1.inv()).magnitude())

            # gripper 差
            d_grip = abs(grip_i - grip_last)

            if (d_trans > trans_thresh) or (d_rot > rot_thresh) or (d_grip > gripper_thresh):
                triggered[side] = True

        # 如果任一侧触发，采样当前帧 i
        if triggered[0] or triggered[1]:
            key_indices.append(i)
            # 仅更新触发的侧的 last_sample（未触发的侧保留原 last_sample）
            for side in (0, 1):
                if triggered[side]:
                    last_sample[side] = i

    # 保证末帧被保留
    if key_indices[-1] != N - 1:
        key_indices.append(N - 1)

    # 转为全局索引并去重排序
    key_indices = np.array(key_indices, dtype=np.int64)
    global_idx = start_idx + np.unique(key_indices)

    return global_idx

def read_endpose(file_path):
    """读取 HDF5 文件中的 endpose 数据集"""
    with h5py.File(file_path, 'r') as hdf:
        if "endpose" not in hdf:
            raise KeyError(f"该文件中不存在 'endpose' 数据集：{file_path}")
        data = hdf["endpose"][:]  # 读取为 numpy 数组
    print(f"✅ 读取完成: endpose.shape = {data.shape}, dtype = {data.dtype}")
    return data


def explore_hdf5(file_path, preview_values=False, indent=0):
    """递归打印 HDF5 文件中所有 group 和 dataset 的层级结构"""
    def print_attrs(name, obj):
        pad = "  " * indent
        if isinstance(obj, h5py.Group):
            print(f"{pad}📂 Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f"{pad}📄 Dataset: {name} | shape={shape}, dtype={dtype}")
            if preview_values:
                data = obj[()]
                # 仅展示部分数据（防止太大）
                preview = np.array2string(data.flatten()[:10], precision=4, separator=", ")
                print(f"{pad}   preview: {preview} ...")

    with h5py.File(file_path, 'r') as hdf:
        print(f"\n📘 Exploring HDF5 file: {file_path}")
        hdf.visititems(print_attrs)

episode = 10
# 输入视频路径

task = "beat_block_hammer_loop" 
# task = "cut_carrot_knife"
# task = "shake_bottle_loop"
# task = "double_knife_chop"
# task = "grab_roller_loop"
input_video = f"/home/liaohaoran/code/RoboTwin/data/{task}/loop1-8-all/video/episode{episode}.mp4"
data_path = f"/home/liaohaoran/code/RoboTwin/data/{task}/loop1-8-all/data/episode{episode}.hdf5"

data_key = "endpose"

data = read_endpose(data_path)

# data_dict = read_hdf5(data_path, data_key)

# data = np.concatenate(data, axis=1)



output = adaptive_fixthre_endpose_sample_indices(0, len(data), data)
print(output.shape)
print(output)

# 输出视频路径
output_video = f"/home/liaohaoran/code/RoboTwin/eval_result/test_sampling/{task}_video_{episode}.mp4"
ori_video = f"/home/liaohaoran/code/RoboTwin/eval_result/test_sampling/{task}_video_{episode}_ori.mp4"

# 目标帧列表（0-based索引，若原列表是1-based需先减1）
target_frames = output
target_frames.sort()  # 确保帧按顺序排列


# # 打开视频并获取基本信息
# cap = cv2.VideoCapture(input_video)
# if not cap.isOpened():
#     print("无法打开视频文件")
#     exit()

# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽度
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高度
# fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数


# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(ori_video, fourcc, fps, (frame_width, frame_height))

# # 4. 循环读取并写入帧
# while cap.isOpened():
#     ret, frame = cap.read()  # ret：是否读取到帧；frame：帧数据
#     if not ret:
#         break  # 无帧可读时退出循环
#     out.write(frame)  # 写入当前帧到输出视频

# # 5. 释放资源
# cap.release()  # 关闭输入视频读取
# out.release()  # 关闭输出视频写入
# cv2.destroyAllWindows()  # 关闭可能打开的窗口

# print(f"视频已保存至：{ori_video}")




fps = 5  # 新视频的帧率（可自定义，如每秒播放2帧）

# 打开视频并获取基本信息
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高度
fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

# 设置编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

current_frame = 0
extracted_count = 0  # 已提取的帧数

while cap.isOpened() and extracted_count < len(target_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    # 检查当前帧是否是目标帧
    if current_frame == target_frames[extracted_count]:
        out.write(frame)  # 写入新视频
        extracted_count += 1  # 移动到下一个目标帧
    
    current_frame += 1

# 释放资源
cap.release()
out.release()


print(f"抽帧合成视频完成：{output_video}")
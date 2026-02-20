import cv2
import os
import h5py
import numpy as np

def read_hdf5(file_path, group_name):
    """
    读取HDF5文件并展示其结构和内容
    
    参数:
        file_path: HDF5文件的路径
    """
    # 打开HDF5文件（'r'表示只读模式）
    group_info = {"type": "group", "datasets": {}}

    with h5py.File(file_path, 'r') as hdf:
        if group_name not in hdf or not isinstance(hdf[group_name], h5py.Group):
            raise ValueError(f"文件中不存在组: {group_name}")
        group = hdf[group_name]
        # 遍历组内所有数据集
        for ds_name, ds in group.items():
            if isinstance(ds, h5py.Dataset):
                group_info["datasets"][ds_name] = ds[:]
    return {group_name: group_info}


def adaptive_joint_sample_indices(
    start_idx: int,
    end_idx: int,
    traj: list[np.ndarray],
    dist_thresh: float = 0.1
) -> np.ndarray:
    """
    向量化版本：基于累计距离的自适应采样（自动决定采样数量）
    使用 np.cumsum + 分段检测，避免Python for循环。

    Args:
        start_idx (int): 起始索引（包含）
        end_idx (int): 结束索引（包含）
        traj (list[np.ndarray]): 每帧的机械臂状态
        dist_thresh (float): 累积距离阈值，越小采样越密

    Returns:
        np.ndarray: 自动采样得到的帧索引，dtype=int64
    """
    assert 0 <= start_idx < end_idx <= traj.shape[0], "索引范围非法"

    # === Step 1: 计算相邻帧间距离 ===
    eps = 1e-8  # 避免除零
    traj_np = np.stack(traj[start_idx:end_idx])  # shape [L, C]
    min_vals = traj_np.min(axis=0, keepdims=True)  # (1, C)，每个通道的最小值
    max_vals = traj_np.max(axis=0, keepdims=True)  # (1, C)，每个通道的最大值
    traj_norm = (traj_np - min_vals) / (max_vals - min_vals + eps)  # (L-1, C)

    diffs = np.abs(traj_norm[1:] - traj_norm[:-1])
    dists = diffs.max(axis=1)

    # === Step 2: 累计距离 & 分组 ===
    cumsum = np.cumsum(dists)
    group_id = np.floor(cumsum / dist_thresh).astype(np.int64)

    # === Step 3: 找到每个组第一次出现的位置 ===
    # 组号变化处即为新的采样点
    change_points = np.flatnonzero(np.diff(group_id, prepend=-1))  # prepend=-1 确保首帧被检测

    # 采样的全局索引
    idx = start_idx + change_points
    if len(idx) == 0 or idx[-1] != end_idx-1:
        idx = np.concatenate([idx, [end_idx-1]])

    return idx.astype(np.int64)

episode = 10
# 输入视频路径

task = "beat_block_hammer_loop" 
input_video = f"/home/liaohaoran/code/RoboTwin/data/{task}/loop1-8-counter_blhl/video/episode{episode}.mp4"
data_path = f"/home/liaohaoran/code/RoboTwin/data/{task}/loop1-8-counter_blhl/data/episode{episode}.hdf5"

# task = "cut_carrot_knife"
# input_video = f"/home/liaohaoran/code/RoboTwin/data/{task}/loop1-8-counter_cck/video/episode{episode}.mp4"
# data_path = f"/home/liaohaoran/code/RoboTwin/data/{task}/loop1-8-counter_cck/data/episode{episode}.hdf5"

# task = "shake_bottle_loop"
# input_video = f"/home/liaohaoran/code/RoboTwin/data/{task}/loop1-8-counter_sbl/video/episode{episode}.mp4"
# data_path = f"/home/liaohaoran/code/RoboTwin/data/{task}/loop1-8-counter_sbl/data/episode{episode}.hdf5"

data_key = "joint_action"
data_dict = read_hdf5(data_path, data_key)

data = []
for key, value in data_dict[data_key]["datasets"].items():
    if key not in ["left_arm", "left_gripper", "right_arm", "right_gripper"]:
        continue
    data.append(value.reshape(-1, 1) if value.ndim == 1 else value) 

data = np.concatenate(data, axis=1)
print(data.shape)

# 打开视频并获取基本信息
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高度
fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

assert total_frames == data.shape[0]

output = adaptive_joint_sample_indices(0, len(data), data)
print(output.shape)
print(output)

# 输出视频路径
output_video = f"/home/liaohaoran/code/RoboTwin/eval_result/test_sampling/{task}_video_{episode}.mp4"

# 目标帧列表（0-based索引，若原列表是1-based需先减1）
target_frames = output

target_frames.sort()  # 确保帧按顺序排列


fps = 5  # 新视频的帧率（可自定义，如每秒播放2帧）

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
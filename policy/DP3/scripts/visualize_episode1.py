import os
import numpy as np
import plotly.graph_objects as go
import cv2
import zarr

# 配置路径
data_path = "/home/liaohaoran/code/RoboTwin/policy/DP3/data/bbhl_demogen-loop1-5-200.zarr/data/point_cloud"
episode_ends_path = "/home/liaohaoran/code/RoboTwin/policy/DP3/data/bbhl_demogen-loop1-5-200.zarr/meta/episode_ends"

# 更新加载 episode_ends 的函数
def load_episode_ends(episode_ends_path):
    if not os.path.exists(episode_ends_path):
        raise FileNotFoundError(f"路径不存在: {episode_ends_path}")

    z = zarr.open(episode_ends_path, mode='r')
    episode_ends = z[:]
    if len(episode_ends) == 0:
        raise ValueError("episode_ends 数据为空！")

    return episode_ends

# 修复 Zarr 数据集打开逻辑，支持直接打开数组
def load_point_cloud(data_path, start_frame, end_frame):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"路径不存在: {data_path}")

    print(f"尝试打开 Zarr 数据集: {data_path}")
    try:
        z = zarr.open(data_path, mode='r')  # 尝试直接打开数组
        print(f"成功打开 Zarr 数组，形状: {z.shape}, 数据类型: {z.dtype}")
    except zarr.errors.ContainsArrayError:
        raise ValueError(f"路径 {data_path} 指向的是一个数组，而不是一个组！请检查路径是否正确。")

    if z.shape[0] == 0:
        raise ValueError("点云数据为空！")

    return np.concatenate(z[start_frame:end_frame], axis=0)

# 将点云数据转换为 Plotly 可视化
def visualize_point_cloud(xyz, rgb=None, output_video="output.mp4", frame_rate=5):
    if rgb is None:
        colors = "rgb(150,150,150)"
    else:
        colors = rgb_array_to_plotly_colors(rgb)

    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    num_frames = xyz.shape[0] // frame_rate  # 按帧率计算帧数
    for i in range(num_frames):
        frame_xyz = xyz[i * frame_rate:(i + 1) * frame_rate]
        frame_rgb = rgb[i * frame_rate:(i + 1) * frame_rate] if rgb is not None else None

        scatter = go.Scatter3d(
            x=frame_xyz[:, 0], y=frame_xyz[:, 1], z=frame_xyz[:, 2],
            mode='markers',
            marker=dict(
                size=1.5,
                symbol='circle',
                color=colors,
                line=dict(width=0),
                opacity=1.0
            )
        )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            title=f"Point Cloud Frame {i + 1}"
        )

        fig = go.Figure(data=[scatter], layout=layout)
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        fig.write_image(frame_path)

    # 合成视频
    frame_files = sorted(os.listdir(temp_dir))
    first_frame = cv2.imread(os.path.join(temp_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(temp_dir, frame_file))
        video_writer.write(frame)

    video_writer.release()
    print(f"视频已保存到 {output_video}")

    # 清理临时文件
    for frame_file in frame_files:
        os.remove(os.path.join(temp_dir, frame_file))
    os.rmdir(temp_dir)

# 更新 save_point_cloud_video 函数，添加检查逻辑
def save_point_cloud_video(fig, output_video, frame_rate):
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # 检查是否有帧数据
    if not hasattr(fig, 'frames') or len(fig.frames) == 0:
        raise ValueError("图形对象没有帧数据，无法生成视频！")

    # 导出每一帧
    for i, frame in enumerate(fig.frames):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        fig.write_image(frame_path)

    # 检查是否成功生成帧文件
    frame_files = sorted(os.listdir(temp_dir))
    if len(frame_files) == 0:
        raise RuntimeError("未能生成任何帧文件，请检查图形对象或导出逻辑！")

    # 读取第一帧以获取视频尺寸
    first_frame = cv2.imread(os.path.join(temp_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    # 初始化视频写入器
    video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(temp_dir, frame_file))
        video_writer.write(frame)

    video_writer.release()
    print(f"视频已保存到 {output_video}")

    # 清理临时文件
    for frame_file in frame_files:
        os.remove(os.path.join(temp_dir, frame_file))
    os.rmdir(temp_dir)

# 将 RGB 数组转换为 Plotly 可接受的颜色格式
def rgb_array_to_plotly_colors(rgb_arr):
    rgb_arr = np.asarray(rgb_arr, dtype=float)
    if rgb_arr.max() <= 1.0:
        rgb_arr = np.clip(rgb_arr * 255.0, 0, 255)
    else:
        rgb_arr = np.clip(rgb_arr, 0, 255)
    rgb_int = rgb_arr.astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb_int]

if __name__ == "__main__":
    # 加载 episode_ends 数据
    episode_ends = load_episode_ends(episode_ends_path)

    # 获取 episode1 的起始和终止帧
    start_frame = 0
    end_frame = episode_ends[0]

    # 加载 episode1 的点云数据
    point_cloud = load_point_cloud(data_path, start_frame, end_frame)

    # 分离 xyz 和 rgb
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:6] if point_cloud.shape[1] >= 6 else None

    # 可视化并保存为视频
    visualize_point_cloud(xyz, rgb, output_video="episode1_point_cloud.mp4")
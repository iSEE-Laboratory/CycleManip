# save_pointcloud_html.py
import os
import numpy as np

# 如果你已经在内存里有 points（N,6），就把它注释掉下面的加载部分
# points = np.hstack((points_world_np, points_color_np))

def visualize_point_cloud_as_video(zarr_path, output_video_path, fps=5):
    """
    可视化 Zarr 数据集中的点云并保存为视频。

    参数:
        zarr_path: Zarr 数据集路径
        output_video_path: 输出视频路径
        fps: 视频帧率
    """
    # 加载点云数据
    point_cloud_path = os.path.join(zarr_path, "data/point_cloud")
    point_cloud = zarr.open(point_cloud_path, mode='r')

    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_size = (640, 480)  # 视频分辨率
    out = cv2.VideoWriter(output_video_path, fourcc, fps, video_size)

    # 遍历每一帧点云
    for i in range(point_cloud.shape[0]):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 黑色背景

        # 获取当前帧点云
        points = point_cloud[i]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # 创建 3D 图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # 保存为图像
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.resize(img, video_size)
        out.write(img)

        plt.close(fig)

    out.release()
    print(f"视频已保存到: {output_video_path}")

# 否则脚本会尝试加载当前目录下的 points.npy
dir = "/home/liaohaoran/code/RoboTwin/points.npy"
if 'points' not in globals():
    if os.path.exists(dir):
        points = np.load(dir)
        print(f"Loaded points from {dir}")
    else:
        raise FileNotFoundError(f"找不到变量 `points` 或文件 {dir}。请把 points 变量注入或先运行保存步骤。")

# 检查 shape
if points.ndim != 2 or points.shape[1] < 3:
    raise ValueError(f"points 应为形状 (N, >=3)，当前为 {points.shape}")

# 支持 (N,3) / (N,4) / (N,6) 等：优先取前三列为 xyz，后面三列（如果存在）为 rgb
xyz = points[:, :3]
rgb = None
if points.shape[1] >= 6:
    rgb = points[:, 3:6]
elif points.shape[1] == 4:
    # 若第四列是灰度值，扩成 RGB
    gray = points[:, 3:4]
    rgb = np.repeat(gray, 3, axis=1)

# 处理颜色：转换为 Plotly 可接受的 'rgb(r,g,b)' 字符串列表
def rgb_array_to_plotly_colors(rgb_arr):
    # rgb_arr shape: (N,3), values either in [0,1] or [0,255]
    rgb_arr = np.asarray(rgb_arr, dtype=float)
    if rgb_arr.max() <= 1.0:
        rgb_arr = np.clip(rgb_arr * 255.0, 0, 255)
    else:
        rgb_arr = np.clip(rgb_arr, 0, 255)
    # 转为整数并生成字符串
    rgb_int = rgb_arr.astype(int)
    return [f"rgb({r},{g},{b})" for r,g,b in rgb_int]

if rgb is None:
    # 若没有颜色，用灰色
    colors = "rgb(150,150,150)"
else:
    colors = rgb_array_to_plotly_colors(rgb)

# 创建交互式 HTML（使用 plotly）
try:
    import plotly.graph_objects as go

    scatter = go.Scatter3d(
        x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
        mode='markers',
        marker=dict(
            size=1.5,          # 点大小，可调整
            symbol='circle',
            color=colors,      # 支持单一颜色或每点颜色列表
            line=dict(width=0),
            opacity=1.0
        )
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'  # 保持比例 (不变形)
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Point Cloud (interactive)"
    )

    fig = go.Figure(data=[scatter], layout=layout)

    # 输出文件名
    out_html = "pointcloud.html"
    # include_plotlyjs='cdn' 会在 HTML 中引用线上 plotly.js (文件更小)，
    # 如果需要完全离线（无需网络），把 include_plotlyjs=True。
    fig.write_html(out_html, include_plotlyjs='cdn', full_html=True)
    print(f"Saved interactive point cloud to {out_html}")
except ImportError:
    raise ImportError("需要安装 plotly: pip install plotly")

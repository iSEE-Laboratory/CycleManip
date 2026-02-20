"""
相机参数实际应用示例
展示如何在实际代码中获取和使用相机参数
"""

import numpy as np
import yaml
import os
import math


# ==================== 示例 1: 直接计算相机内参 ====================

def get_camera_intrinsic_from_config():
    """
    示例: 从配置文件直接计算相机内参
    
    这是最直接的方式，不需要加载整个环境
    """
    
    # 读取配置
    config_path = "task_config/_camera_config.yml"
    with open(config_path, 'r') as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # 获取 D435 配置
    d435_config = camera_config['D435']
    
    fovy_deg = d435_config['fovy']
    width = d435_config['w']
    height = d435_config['h']
    
    # 计算内参
    fovy_rad = math.radians(fovy_deg)
    fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy
    ppx = width / 2.0
    ppy = height / 2.0
    
    # 构建内参矩阵
    K = np.array([
        [fx,  0, ppx],
        [ 0, fy, ppy],
        [ 0,  0,   1]
    ], dtype=np.float32)
    
    print("D435 内参矩阵:")
    print(K)
    print(f"\nfx={fx:.4f}, fy={fy:.4f}")
    print(f"ppx={ppx:.4f}, ppy={ppy:.4f}")
    
    return K, (fx, fy, ppx, ppy)


# ==================== 示例 2: 在环境中获取相机参数 ====================

def get_camera_params_from_environment():
    """
    示例: 从已初始化的环境获取相机参数
    
    这需要环境已经创建好相机对象
    """
    
    # 假设环境已创建
    # task = Base_Task(...)
    # camera_system = task.cameras
    
    # 获取相机配置（这会调用 SAPIEN 的 get_intrinsic_matrix 等方法）
    # camera_config = camera_system.get_config()
    
    # 各种相机的参数
    # for camera_name, params in camera_config.items():
    #     intrinsic = params['intrinsic_cv']      # 3x3 内参矩阵
    #     extrinsic = params['extrinsic_cv']      # 外参矩阵
    #     cam2world = params['cam2world_gl']      # 4x4 相机到世界变换
    #     
    #     print(f"{camera_name}:")
    #     print(f"  内参:\n{intrinsic}")
    #     print(f"  外参:\n{extrinsic}")
    #     print(f"  相机到世界:\n{cam2world}")
    
    pass


# ==================== 示例 3: 深度值处理 ====================

def process_depth_values(depth_image, depth_scale=0.001):
    """
    示例: 处理从 SAPIEN 获取的深度值
    
    Args:
        depth_image: 从 camera.get_picture('Position') 获取的深度
        depth_scale: 深度缩放因子 (mm 到 m)
    
    Returns:
        depth_in_meters: 以米为单位的深度
    """
    
    # SAPIEN 的深度是以米为单位，通常转换为毫米
    depth_in_mm = depth_image * 1000.0
    
    # 如果需要转回米
    depth_in_meters = depth_in_mm * depth_scale
    
    # 处理无效深度 (e.g., 超出范围)
    valid_mask = (depth_in_meters > 0.1) & (depth_in_meters < 100)
    depth_cleaned = depth_in_meters.copy()
    depth_cleaned[~valid_mask] = 0
    
    return depth_cleaned


# ==================== 示例 4: 点到像素坐标的投影 ====================

def project_3d_point_to_image(point_3d, intrinsic_matrix, extrinsic_matrix):
    """
    示例: 将 3D 世界坐标点投影到图像坐标
    
    Args:
        point_3d: 3D 世界坐标 (3,)
        intrinsic_matrix: 内参矩阵 K (3,3)
        extrinsic_matrix: 外参矩阵 [R|t] (3,4) 或完整 (4,4)
    
    Returns:
        pixel_coord: 像素坐标 (u, v)
    """
    
    # 转换为齐次坐标
    point_homogeneous = np.append(point_3d, 1.0)
    
    # 相机坐标 = 外参 @ 世界坐标
    if extrinsic_matrix.shape[0] == 4:
        point_camera = extrinsic_matrix @ point_homogeneous
    else:  # 3x4 形式
        point_camera = extrinsic_matrix @ point_homogeneous
    
    # 检查点是否在相机前方
    if point_camera[2] <= 0:
        return None  # 点在相机后方
    
    # 投影到图像平面
    point_pixel = intrinsic_matrix @ point_camera[:3]
    u = point_pixel[0] / point_pixel[2]
    v = point_pixel[1] / point_pixel[2]
    
    return (u, v)


# ==================== 示例 5: 像素坐标反投影到 3D ====================

def back_project_pixel_to_3d(u, v, depth, intrinsic_matrix, extrinsic_matrix):
    """
    示例: 将像素坐标和深度值反投影回 3D 世界坐标
    
    Args:
        u, v: 像素坐标
        depth: 深度值（米）
        intrinsic_matrix: 内参矩阵 K (3,3)
        extrinsic_matrix: 外参矩阵 (3,4) 或 (4,4)
    
    Returns:
        point_3d: 3D 世界坐标
    """
    
    # 计算内参矩阵的逆
    K_inv = np.linalg.inv(intrinsic_matrix)
    
    # 像素齐次坐标
    pixel_homogeneous = np.array([u, v, 1.0])
    
    # 相机坐标（深度方向标准化后乘以实际深度）
    point_camera = K_inv @ pixel_homogeneous * depth
    
    # 相机齐次坐标
    point_camera_homo = np.append(point_camera, 1.0)
    
    # 反演外参矩阵以从相机坐标转到世界坐标
    if extrinsic_matrix.shape[0] == 4:
        ext_inv = np.linalg.inv(extrinsic_matrix)
        point_world = ext_inv @ point_camera_homo
        return point_world[:3]
    else:  # 3x4 形式
        # [R|t] 的逆: [R^T | -R^T @ t]
        R = extrinsic_matrix[:, :3]
        t = extrinsic_matrix[:, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        
        # 构建 4x4 变换矩阵
        ext_inv = np.eye(4)
        ext_inv[:3, :3] = R_inv
        ext_inv[:3, 3] = t_inv
        
        point_world = ext_inv @ point_camera_homo
        return point_world[:3]


# ==================== 示例 6: 获取相机位置和朝向 ====================

def extract_camera_pose_from_matrix(model_matrix):
    """
    示例: 从相机到世界的变换矩阵中提取相机位置和朝向
    
    Args:
        model_matrix: 4x4 相机到世界变换矩阵
    
    Returns:
        position: 相机在世界坐标系中的位置 (3,)
        forward: 相机前方方向向量 (3,)
        up: 相机上方方向向量 (3,)
        right: 相机右方方向向量 (3,)
    """
    
    # 提取旋转矩阵
    R = model_matrix[:3, :3]
    
    # 提取平移向量（相机位置）
    position = model_matrix[:3, 3]
    
    # 相机方向向量
    # 通常相机的 -Z 轴是前方方向
    forward = -R[:, 2]  # 第三列的负
    up = R[:, 1]        # 第二列是上方
    right = R[:, 0]     # 第一列是右方
    
    # 标准化
    forward = forward / np.linalg.norm(forward)
    up = up / np.linalg.norm(up)
    right = right / np.linalg.norm(right)
    
    return position, forward, up, right


# ==================== 示例 7: 比较模拟和真实相机参数 ====================

def compare_simulated_vs_real_camera():
    """
    示例: 比较模拟 SAPIEN D435 和真实 RealSense D435
    """
    
    print("=" * 60)
    print("D435 相机参数对比")
    print("=" * 60)
    
    # 模拟相机（从配置计算）
    fovy_deg = 37
    w_sim, h_sim = 320, 240
    fovy_rad = math.radians(fovy_deg)
    fy_sim = (h_sim / 2.0) / math.tan(fovy_rad / 2.0)
    fx_sim = fy_sim
    
    print("\n[模拟 SAPIEN D435 (320×240)]")
    print(f"  fovy: {fovy_deg}°")
    print(f"  fx: {fx_sim:.2f} px")
    print(f"  fy: {fy_sim:.2f} px")
    print(f"  ppx: {w_sim/2:.2f} px")
    print(f"  ppy: {h_sim/2:.2f} px")
    
    # 真实相机（来自 piper_real_env.py）
    print("\n[真实 RealSense D435 (640×480)]")
    print(f"  fx: 382.59 px")
    print(f"  fy: 382.18 px")
    print(f"  ppx: 323.70 px")
    print(f"  ppy: 240.38 px")
    
    # 计算差异
    print("\n[差异分析]")
    print(f"  焦距差异 (fx): {abs(fx_sim - 382.59):.2f} px ({abs(fx_sim - 382.59)/382.59*100:.2f}%)")
    print(f"  主点差异 (ppx): {abs(w_sim/2 - 323.70):.2f} px")
    print(f"  主点差异 (ppy): {abs(h_sim/2 - 240.38):.2f} px")
    
    print("\n[建议]")
    print("  - 模拟和真实参数不完全相同，但在同一量级")
    print("  - 如果精度要求高，建议在模拟中使用标定的真实参数")
    print("  - 或在数据处理中做参数调整和校准")


# ==================== 示例 8: 完整使用流程 ====================

def complete_workflow_example():
    """
    示例: 完整的相机参数获取和使用流程
    """
    
    print("\n" + "=" * 60)
    print("完整工作流示例")
    print("=" * 60)
    
    # 第一步：获取配置
    print("\n[步骤 1] 读取相机配置...")
    config_path = "task_config/_camera_config.yml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(f"  ✓ 找到 {len(camera_config)} 种相机类型")
    
    # 第二步：计算内参
    print("\n[步骤 2] 计算相机内参...")
    K, (fx, fy, ppx, ppy) = get_camera_intrinsic_from_config()
    print(f"  ✓ 计算完成")
    
    # 第三步：显示内参
    print("\n[步骤 3] 内参矩阵...")
    print(f"  K =\n{K}")
    
    # 第四步：示例投影和反投影
    print("\n[步骤 4] 示例投影和反投影...")
    
    # 假设的 3D 点
    point_3d = np.array([0.5, 0.2, 1.5])  # 世界坐标
    print(f"  3D 点 (世界坐标): {point_3d}")
    
    # 假设的外参（恒等变换用于演示）
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [0, 0, 0]  # 相机在原点
    
    # 投影
    pixel = project_3d_point_to_image(point_3d, K, extrinsic[:3, :])
    if pixel:
        print(f"  投影到像素坐标: {pixel}")
        
        # 反投影
        depth = np.linalg.norm(point_3d)  # 使用欧氏距离作为深度
        back_point = back_project_pixel_to_3d(
            pixel[0], pixel[1], depth, K, extrinsic
        )
        print(f"  反投影回 3D: {back_point}")
        print(f"  误差: {np.linalg.norm(point_3d - back_point):.6f}")
    
    print("\n✓ 工作流完成！")


if __name__ == "__main__":
    # 运行所有示例
    print("相机参数应用示例\n")
    
    # 示例 1
    print("\n" + "="*60)
    print("示例 1: 从配置文件计算内参")
    print("="*60)
    get_camera_intrinsic_from_config()
    
    # 示例 7
    print("\n" + "="*60)
    print("示例 7: 相机参数对比")
    print("="*60)
    compare_simulated_vs_real_camera()
    
    # 示例 8
    print("\n" + "="*60)
    print("示例 8: 完整工作流")
    print("="*60)
    complete_workflow_example()
    
    print("\n" + "="*60)
    print("所有示例完成！")
    print("="*60)

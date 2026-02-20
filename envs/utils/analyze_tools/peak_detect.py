#!/usr/bin/env python3
"""
峰值检测方法 - 简化版本，仅使用PCA降维
对外暴露peak_analysis接口用于检测多维时间序列的峰值
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.decomposition import PCA
from datetime import datetime
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def peak_detect(data, smooth=True, smooth_window=20,height_factor=0.4, distance_factor=30, prominence_factor=0.075, m_shape_peak_factor=0.3, save_plot=False, save_path=""):
    """
    使用PCA降维和峰值检测分析时间序列
    
    Parameters:
    -----------
    data : numpy.ndarray
        输入数据，形状为 (n_samples, n_features)
    smooth : bool, optional
        是否对信号进行平滑处理，默认为True
    smooth_window : int, optional
        平滑窗口大小，默认为20
    height_factor : float, optional
        峰值高度阈值系数，默认为0.5（峰值 = mean + height_factor * std）
    distance_factor : int, optional
        峰值间最小距离系数，默认为20（distance = len(data) // distance_factor）
    prominence_factor : float, optional
        峰值突出度系数，默认为0.1（prominence = prominence_factor * std）

    m_shape_peak_factor : float, optional
        用于防止M字形峰值误检的系数，默认为0.3
        检测方式为：如果两个峰之间的 最小值 到 较小峰值的高度 差值 大于 m_shape_peak_factor * avg_diff，则认为是m字形误检，去掉较小峰值。

    save_plot : bool, optional
        是否保存可视化图片，默认为False
    save_path : str, optional
        图片保存路径，必须显示指定，默认为空字符串，推荐这样
        f"{self.eval_video_path}/episode{self.test_num}.png"
    
    Returns:
    --------
    num_peaks : int
        检测到的峰值个数
    peak_positions : list
        每个峰值的时间点（索引位置）列表
    
    Example:
    --------
    >>> data = np.load('your_data.npy')
    >>> num_peaks, peak_positions = peak_analysis(data, save_plot=True)
    >>> print(f"检测到 {num_peaks} 个峰值")
    >>> print(f"峰值位置: {peak_positions}")
    """
    
    # 数据验证
    if not isinstance(data, np.ndarray):
        raise TypeError("输入数据必须是numpy数组")
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.ndim != 2:
        raise ValueError(f"输入数据必须是二维数组，当前维度: {data.ndim}")
    
    if data.shape[0] == 0:
        return 0, []
    
    if data.shape[0] < 3:
        raise ValueError(f"数据样本数太少，至少需要3个样本，当前样本数: {data.shape[0]}")
    
    # 数据预处理：标准化
    data_processed = data.copy()
    # 处理常数列（标准差为0的情况）
    std_vals = np.std(data_processed, axis=0)
    std_vals[std_vals == 0] = 1.0  # 避免除以0
    data_processed = (data_processed - np.mean(data_processed, axis=0)) / std_vals
    
    signal_1d = signal_1d_smooth = data[:, 0]  # 直接使用第一维数据，避免PCA不稳定性
    
    # 平滑处理
    if smooth and len(signal_1d) > 5:
        # 确定窗口长度（必须是奇数且>=3）
        window_length = min(smooth_window, len(signal_1d) if len(signal_1d) % 2 == 1 else len(signal_1d) - 1)
        if window_length >= 3:
            signal_1d_smooth = savgol_filter(signal_1d, window_length, 2)
        else:
            signal_1d_smooth = signal_1d
    else:
        signal_1d_smooth = signal_1d
    
    # 峰值检测参数设置
    height = np.mean(signal_1d_smooth) + height_factor * np.std(signal_1d_smooth)
    distance = max(1, len(signal_1d) // distance_factor)
    prominence = prominence_factor * np.std(signal_1d_smooth)
    
    # 峰值检测
    peaks, properties = find_peaks(
        signal_1d_smooth,
        height=height,
        distance=distance,
        prominence=prominence
    )
    
    # 返回结果
    num_peaks = len(peaks)
    peak_positions = peaks.tolist()

    # peak距离小于10的，认为是同一个峰值，去掉
    if num_peaks > 1:
        filtered_peaks = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered_peaks[-1] >= 10:
                filtered_peaks.append(p)
        peak_positions = filtered_peaks
        num_peaks = len(peak_positions)

    ##################### 防止 M 字形峰值误检####################

    # 检查相邻峰中间的最小值,拿到所有峰之间的最小值的数组
    min_between_peaks = []
    if num_peaks > 1:
        for i in range(len(peak_positions) - 1):
            start = peak_positions[i]
            end = peak_positions[i + 1]
            if end - start > 1:
                min_val = np.min(signal_1d_smooth[start:end])
                min_between_peaks.append(min_val)
            else:
                min_between_peaks.append(signal_1d_smooth[start])
    
    # 拿到峰的平均值
    avg_peak_value = np.mean(signal_1d_smooth[peak_positions]) if num_peaks > 0 else 0
    avg_min_between = np.mean(min_between_peaks) if len(min_between_peaks) > 0 else 0
    avg_diff = avg_peak_value - avg_min_between
    # 如果两个峰之间的最小值大于 avg_peak_value - 0.3 * avg_diff, 则认为是误检，去掉更小的那个峰，保留更大的那个峰
    if num_peaks > 1 and avg_diff > 0:
        refined_peaks = [peak_positions[0]]
        for i in range(1, len(peak_positions)):
            start = peak_positions[i - 1]
            end = peak_positions[i]
            if end - start > 1:
                min_val = np.min(signal_1d_smooth[start:end])
                min_peak_val = min(signal_1d_smooth[start], signal_1d_smooth[end])
                if abs(min_val - min_peak_val) >= m_shape_peak_factor * avg_diff:
                    # 正常情况，保留当前峰
                    refined_peaks.append(peak_positions[i])
                else:
                    # 误检情况，保留更大的峰
                    curr_peak_val = signal_1d_smooth[peak_positions[i]]
                    prev_peak_val = signal_1d_smooth[peak_positions[i - 1]]
                    if curr_peak_val > prev_peak_val:
                        refined_peaks[-1] = peak_positions[i]
            else:
                refined_peaks.append(peak_positions[i])
        peak_positions = refined_peaks
        num_peaks = len(peak_positions)

    ##################### 防止 M 字形峰值误检####################

    # 如果需要保存图片
    if save_plot:
        _save_peak_plot(data, signal_1d, signal_1d_smooth, peak_positions, save_path)
    
    return num_peaks, peak_positions


def _save_peak_plot(data, signal_1d, signal_1d_smooth, peaks, save_path):
    """
    绘制并保存峰值检测结果图
    
    Parameters:
    -----------
    data : numpy.ndarray
        原始多维数据
    signal_1d : numpy.ndarray
        PCA降维后的一维信号
    signal_1d_smooth : numpy.ndarray
        平滑后的一维信号
    peaks : numpy.ndarray
        检测到的峰值索引
    """
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 1. 原始多维数据
    axes[0].set_title('Original Multi-dimensional Data', fontsize=12, fontweight='bold')
    num_dims = min(data.shape[1], 10)  # 最多显示10个维度
    for i in range(num_dims):
        axes[0].plot(data[:, i], label=f'Dim {i+1}', alpha=0.7, linewidth=1.5)
    if data.shape[1] > num_dims:
        axes[0].text(0.02, 0.98, f'(showing {num_dims} dimensions, total {data.shape[1]} dimensions)', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].legend(loc='upper right', ncol=min(5, num_dims))
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    
    # 2. PCA降维后的信号和检测到的峰值
    axes[1].set_title(f'1D Signal (PCA) with Detected Peaks (Total: {len(peaks)} peaks)', 
                     fontsize=12, fontweight='bold')
    axes[1].plot(signal_1d, 'b-', alpha=0.5, label='Original Signal', linewidth=1)
    axes[1].plot(signal_1d_smooth, 'g-', linewidth=2, label='Smoothed Signal')
    axes[1].plot(peaks, signal_1d_smooth[peaks], 'ro', markersize=10, 
                label=f'Detected Peaks ({len(peaks)})', zorder=5)
    
    # 标注峰值位置
    for i, peak in enumerate(peaks):
        axes[1].annotate(f'{peak}', xy=(peak, signal_1d_smooth[peak]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('PCA Component Value')
    
    # 3. 峰值间隔分析
    if len(peaks) > 1:
        intervals = np.diff(peaks)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        axes[2].set_title(f'Peak Intervals (Avg: {avg_interval:.2f} ± {std_interval:.2f})', 
                         fontsize=12, fontweight='bold')
        bars = axes[2].bar(range(len(intervals)), intervals, alpha=0.7, 
                          color='steelblue', edgecolor='black', linewidth=1.5)
        axes[2].axhline(y=avg_interval, color='r', linestyle='--', linewidth=2,
                       label=f'Average: {avg_interval:.2f}')
        axes[2].axhline(y=avg_interval + std_interval, color='orange', 
                       linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1 Std')
        axes[2].axhline(y=avg_interval - std_interval, color='orange', 
                       linestyle=':', linewidth=1.5, alpha=0.7)
        
        # 标注每个间隔值
        for i, (bar, interval) in enumerate(zip(bars, intervals)):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(interval)}', ha='center', va='bottom', fontsize=9)
        
        axes[2].set_xlabel('Peak Pair Index')
        axes[2].set_ylabel('Interval (time steps)')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3, axis='y')
    else:
        axes[2].text(0.5, 0.5, 'Insufficient peaks for interval analysis\n(at least 2 peaks needed)', 
                    ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[2].set_title('Peak Intervals - Insufficient Data', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图片已保存到: {save_path}")
    
    return save_path


if __name__ == "__main__":
    import time
    base_dir = '/home/haoran/tmp/traj/sbl/'  # 修改为你存放npy文件的目录
    # all npy files
    npy_files = [f for f in os.listdir(base_dir) if f.endswith('.npy')]
    npy_files.sort()
    print(f"找到 {len(npy_files)} 个npy文件")
    for npy_file in npy_files:
        file_path = os.path.join(base_dir, npy_file)
        print(f"处理文件: {file_path}")
        data = np.load(file_path)
        print(f"数据形状: {data.shape}")
        num_peaks, peak_positions = peak_detect(data, save_plot=True, save_path=base_dir+"img/"f"{npy_file.split('.')[0]}.png")
        print(f"检测到 {num_peaks} 个峰值，位置: {peak_positions}")

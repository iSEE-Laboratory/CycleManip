#!/usr/bin/env python3
"""
跃迁检测方法 - 检测时间序列中的大幅变化
专门用于检测在时间窗口内是否有大幅跃迁（变化幅度超过阈值）
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def transition_detect(data, window_size=50, threshold=0.6, smooth=True, 
                     min_interval=100, save_plot=False):
    """
    检测时间序列中的大幅跃迁
    
    Parameters:
    -----------
    data : numpy.ndarray
        输入数据，形状为 (n_samples,)
    window_size : int, optional
        检测窗口大小，默认为50
    threshold : float, optional
        跃迁幅度阈值，默认为0.6
    smooth : bool, optional
        是否对信号进行平滑处理，默认为True
    min_interval : int, optional
        最小跃迁间隔，避免重复检测，默认为100
    save_plot : bool, optional
        是否保存可视化图片，默认为False
    
    Returns:
    --------
    num_transitions : int
        检测到的大幅跃迁个数
    transition_positions : list
        每个跃迁的位置（窗口中心点）列表
    transition_magnitudes : list
        每个跃迁的幅度列表
    
    Example:
    --------
    >>> data = np.load('your_data.npy')
    >>> num_trans, positions, magnitudes = transition_detect(data, save_plot=True)
    >>> print(f"检测到 {num_trans} 个大幅跃迁")
    >>> print(f"跃迁位置: {positions}")
    >>> print(f"跃迁幅度: {magnitudes}")
    """
    
    # 数据验证
    if not isinstance(data, np.ndarray):
        raise TypeError("输入数据必须是numpy数组")
    
    if data.size == 0:
        return 0, [], []
    
    if data.ndim != 1:
        raise ValueError("输入数据必须是一维数组")
    
    signal = data.copy()
    
    # 平滑处理
    if smooth and len(signal) > 5:
        window_length = min(21, len(signal))
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(5, window_length)
        
        if window_length >= 5:
            signal_smooth = savgol_filter(signal, window_length, 2)
        else:
            signal_smooth = signal
    else:
        signal_smooth = signal
    
    # 跃迁检测
    transitions = []
    magnitudes = []
    last_transition_pos = -min_interval  # 确保第一个跃迁能被检测到
    
    # 滑动窗口检测
    for i in range(0, len(signal_smooth) - window_size + 1, window_size // 2):
        # 跳过距离上次跃迁太近的窗口
        if i - last_transition_pos < min_interval:
            continue
            
        window = signal_smooth[i:i + window_size]
        window_min = np.min(window)
        window_max = np.max(window)
        magnitude = abs(window_max - window_min)
        
        # 检查是否超过阈值
        if magnitude > threshold:
            # 找到跃迁发生的精确位置（窗口内变化最大的点）
            window_center = i + window_size // 2
            
            # 检查是否与之前的跃迁有足够间隔
            if window_center - last_transition_pos >= min_interval:
                transitions.append(window_center)
                magnitudes.append(magnitude)
                last_transition_pos = window_center
    
    # 返回结果
    num_transitions = len(transitions)
    
    # 如果需要保存图片
    if save_plot:
        _save_transition_plot(signal, signal_smooth, transitions, magnitudes, 
                             window_size, threshold)
    
    return num_transitions, transitions, magnitudes


def _save_transition_plot(signal: np.ndarray, signal_smooth: np.ndarray, 
                         transitions: list, magnitudes: list, 
                         window_size: int, threshold: float):
    """
    绘制并保存跃迁检测结果图
    """
    # 创建保存目录
    save_dir = '/home/haoran/tmp'
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成时间戳文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'transition_{timestamp}.png')
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # 1. 原始信号和平滑信号
    axes[0].set_title('Signal with Detected Transitions', fontsize=12, fontweight='bold')
    axes[0].plot(signal, 'b-', alpha=0.5, label='Original Signal', linewidth=1)
    axes[0].plot(signal_smooth, 'g-', linewidth=2, label='Smoothed Signal')
    
    # 标记检测到的跃迁
    for i, (pos, mag) in enumerate(zip(transitions, magnitudes)):
        axes[0].axvline(x=pos, color='red', linestyle='--', alpha=0.7, 
                       label='Transition' if i == 0 else "")
        axes[0].plot(pos, signal_smooth[pos], 'ro', markersize=8, zorder=5)
        
        # 标注跃迁幅度
        axes[0].annotate(f'Δ={mag:.3f}', xy=(pos, signal_smooth[pos]),
                        xytext=(10, 10), textcoords='offset points',
                        ha='left', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Signal Value')
    
    # 2. 跃迁幅度分布
    axes[1].set_title(f'Transition Magnitudes (Threshold: {threshold})', 
                     fontsize=12, fontweight='bold')
    
    if len(magnitudes) > 0:
        bars = axes[1].bar(range(len(magnitudes)), magnitudes, alpha=0.7,
                          color='steelblue', edgecolor='black', linewidth=1.5)
        axes[1].axhline(y=threshold, color='r', linestyle='-', linewidth=2,
                       label=f'Threshold: {threshold}')
        
        # 标注每个跃迁的幅度
        for i, (bar, mag) in enumerate(zip(bars, magnitudes)):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{mag:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 添加统计信息
        avg_mag = np.mean(magnitudes)
        max_mag = np.max(magnitudes)
        axes[1].text(0.02, 0.98, f'Average: {avg_mag:.3f}\nMax: {max_mag:.3f}', 
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    else:
        axes[1].text(0.5, 0.5, 'No transitions detected\n(未检测到跃迁)', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    axes[1].set_xlabel('Transition Index')
    axes[1].set_ylabel('Magnitude')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"跃迁检测图片已保存到: {save_path}")
    return save_path


def find_transition_details(data, transition_positions, window_half_size=25):
    """
    获取跃迁的详细信息
    
    Parameters:
    -----------
    data : numpy.ndarray
        原始数据
    transition_positions : list
        跃迁位置列表
    window_half_size : int, optional
        用于分析跃迁细节的窗口半宽，默认为25
    
    Returns:
    --------
    details : list
        每个跃迁的详细信息字典列表
    """
    details = []
    
    for pos in transition_positions:
        # 确定分析窗口
        start = max(0, pos - window_half_size)
        end = min(len(data), pos + window_half_size)
        
        window_data = data[start:end]
        
        # 计算窗口内的统计信息
        window_min = np.min(window_data)
        window_max = np.max(window_data)
        magnitude = abs(window_max - window_min)
        
        # 找到最小值和最大值的位置（相对位置）
        min_pos_rel = np.argmin(window_data)
        max_pos_rel = np.argmax(window_data)
        
        # 转换为绝对位置
        min_pos = start + min_pos_rel
        max_pos = start + max_pos_rel
        
        # 确定跃迁方向
        direction = "up" if max_pos > min_pos else "down"
        
        details.append({
            'position': pos,
            'magnitude': magnitude,
            'min_value': window_min,
            'max_value': window_max,
            'min_position': min_pos,
            'max_position': max_pos,
            'direction': direction,
            'window_start': start,
            'window_end': end
        })
    
    return details


if __name__ == "__main__":
    path = '/home/haoran/tmp/traj/1760531842.3523853.npy'
    data = np.load(path)
    print(data.shape) # (N,)
    num_trans, positions, magnitudes = transition_detect(data, save_plot=True)
    print(f"检测到 {num_trans} 个大幅跃迁")
    print(f"跃迁位置: {positions}")
    print(f"跃迁幅度: {magnitudes}")
    details = find_transition_details(data, positions)
    for i, detail in enumerate(details):
        print(f"\n跃迁 {i+1} 详情:")
        for k, v in detail.items():
            print(f"  {k}: {v}")
# 统计周期数方法 - 通过阈值切片法统计时间序列的周期数
# 适用于近似周期性但不完全规则的序列
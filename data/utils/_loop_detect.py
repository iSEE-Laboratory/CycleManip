#!/usr/bin/env python3
"""
聚类与分段方法 - 用于检测多维时间序列的周期性
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def sliding_window_segmentation(signal, window_size, step_size=1):
    """
    滑动窗口分段
    """
    segments = []
    positions = []
    
    for i in range(0, len(signal) - window_size + 1, step_size):
        segment = signal[i:i + window_size]
        segments.append(segment)
        positions.append(i)
    
    return np.array(segments), np.array(positions)

def find_optimal_clusters(segments, max_clusters=10):
    """
    寻找最优聚类数
    """
    if len(segments) < 2:
        return 2, [], []
    
    if len(segments) < max_clusters:
        max_clusters = len(segments)
    
    silhouette_scores = []
    inertias = []
    
    for k in range(2, max_clusters + 1):
        if k > len(segments):
            break
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(segments)
        
        if len(np.unique(labels)) > 1:  # 确保有多个聚类
            silhouette_avg = silhouette_score(segments, labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
        else:
            silhouette_scores.append(-1)
            inertias.append(float('inf'))
    
    if silhouette_scores:
        optimal_k = np.argmax(silhouette_scores) + 2
    else:
        optimal_k = 2
    
    return optimal_k, silhouette_scores, inertias

def clustering_analysis(data, method='sum', window_sizes=None):
    """
    使用聚类分析周期
    """
    print(f"使用聚类分析，降维方法: {method}")
    
    # 数据预处理
    data_processed = data.copy()
    data_processed = (data_processed - np.mean(data_processed, axis=0)) / np.std(data_processed, axis=0)
    
    # 降维到一维 - 只保留sum方法
    signal_1d = np.sum(data_processed, axis=1)
    
    if window_sizes is None:
        # 尝试不同的窗口大小
        min_window = max(3, len(signal_1d) // 20)
        max_window = len(signal_1d) // 3
        window_sizes = [min_window, min_window * 2, min_window * 3]
        window_sizes = [w for w in window_sizes if w <= max_window]
    
    best_results = None
    best_score = -1
    best_window_size = window_sizes[0]
    
    results_all_windows = {}
    
    for window_size in window_sizes:
        print(f"\n--- 窗口大小: {window_size} ---")
        
        # 分段
        segments, positions = sliding_window_segmentation(signal_1d, window_size, step_size=window_size//2)
        
        if len(segments) < 4:  # 需要足够的段进行聚类
            print(f"段数太少 ({len(segments)})，跳过此窗口大小")
            continue
        
        # 标准化段
        scaler = StandardScaler()
        segments_scaled = scaler.fit_transform(segments)
        
        # 寻找最优聚类数
        optimal_k, silhouette_scores, inertias = find_optimal_clusters(segments_scaled)
        print(f"最优聚类数: {optimal_k}")
        
        # 执行聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(segments_scaled)
        
        # 计算轮廓系数
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(segments_scaled, labels)
            print(f"轮廓系数: {silhouette_avg:.4f}")
        else:
            silhouette_avg = -1
        
        # 分析聚类结果
        cluster_info = {}
        for i in range(optimal_k):
            cluster_positions = positions[labels == i]
            cluster_segments = segments[labels == i]
            cluster_info[i] = {
                'positions': cluster_positions,
                'segments': cluster_segments,
                'count': len(cluster_positions)
            }
            print(f"聚类 {i}: {len(cluster_positions)} 个段")
        
        # DBSCAN聚类作为对比
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(segments_scaled)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        print(f"DBSCAN聚类数: {n_clusters_dbscan}")
        
        results_all_windows[window_size] = {
            'segments': segments,
            'positions': positions,
            'labels': labels,
            'dbscan_labels': dbscan_labels,
            'cluster_info': cluster_info,
            'silhouette_score': silhouette_avg,
            'optimal_k': optimal_k
        }
        
        # 更新最佳结果
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_window_size = window_size
            best_results = results_all_windows[window_size]
    
    if best_results is None:
        print("没有找到有效的聚类结果")
        return signal_1d, {}, best_window_size
    
    # 分析周期性
    print(f"\n--- 最佳结果分析 (窗口大小: {best_window_size}) ---")
    
    # 尝试从聚类标签序列中找到周期
    label_sequence = best_results['labels']
    cycle_starts = detect_periodic_pattern(best_results['positions'], label_sequence)
    
    # 检测三个阶段
    prep_phase, main_phase, exit_phase = detect_three_phases(
        best_results['positions'], label_sequence, signal_1d
    )
    
    # 估计平均周期长度
    if len(cycle_starts) > 1:
        intervals = np.diff(cycle_starts)
        avg_period = int(np.mean(intervals))
    else:
        avg_period = best_window_size
    
    best_results['cycle_starts'] = cycle_starts
    best_results['avg_period'] = avg_period
    best_results['preparation_phase'] = prep_phase
    best_results['main_loop_phase'] = main_phase
    best_results['exit_phase'] = exit_phase
    
    print(f"检测到的周期起点: {cycle_starts}")
    print(f"平均周期长度: {avg_period}")
    print(f"准备期: {prep_phase}")
    print(f"主循环期: {main_phase}")
    print(f"跳出期: {exit_phase}")
    
    return signal_1d, best_results, best_window_size, results_all_windows

def detect_periodic_pattern(positions, labels):
    """
    从聚类标签序列中检测周期模式
    基于聚类间隔分析的改进版本
    """
    if len(labels) < 4:
        return [0]
    
    # 分析聚类间隔
    cluster_intervals = analyze_cluster_intervals(positions, labels)
    
    # 找到周期性聚类
    periodic_groups = find_periodic_clusters(cluster_intervals)
    
    if not periodic_groups:
        print("未检测到周期性模式，使用传统方法...")
        return detect_periodic_pattern_traditional(positions, labels)
    
    # 使用最好的周期性组来确定周期起点
    best_group = periodic_groups[0]
    periodic_labels = best_group['labels']
    estimated_period = best_group['avg_interval']
    
    # 收集所有周期性聚类的位置
    all_periodic_positions = []
    for label in periodic_labels:
        if label in cluster_intervals:
            cluster_pos = cluster_intervals[label]['positions']
            all_periodic_positions.extend(cluster_pos)
    
    if not all_periodic_positions:
        return [0]
    
    all_periodic_positions = np.sort(all_periodic_positions)
    
    # 基于估计的周期长度来确定周期起点
    cycle_starts = [all_periodic_positions[0]]  # 从第一个周期性位置开始
    
    current_pos = all_periodic_positions[0]
    while current_pos + estimated_period < positions[-1]:
        next_expected = current_pos + estimated_period
        
        # 寻找最接近期望位置的实际周期性位置
        closest_pos = None
        min_distance = float('inf')
        
        for pos in all_periodic_positions:
            if pos > current_pos:  # 只考虑后面的位置
                distance = abs(pos - next_expected)
                if distance < min_distance and distance < estimated_period * 0.3:  # 容忍度30%
                    min_distance = distance
                    closest_pos = pos
        
        if closest_pos is not None:
            cycle_starts.append(closest_pos)
            current_pos = closest_pos
        else:
            # 如果找不到合适的位置，就按估计间隔推进
            current_pos = next_expected
            cycle_starts.append(current_pos)
    
    return cycle_starts

def detect_periodic_pattern_traditional(positions, labels):
    """
    传统的周期检测方法（作为备用）
    """
    cycle_starts = [0]  # 总是从0开始
    
    # 简单方法：寻找标签序列的重复模式
    for pattern_length in range(2, len(labels)//2 + 1):
        pattern = labels[:pattern_length]
        
        # 检查这个模式是否在序列中重复
        repeats = []
        for i in range(0, len(labels) - pattern_length + 1, pattern_length):
            segment = labels[i:i + pattern_length]
            if len(segment) == pattern_length and np.array_equal(segment, pattern):
                repeats.append(i)
        
        if len(repeats) >= 2:  # 至少重复一次
            # 将索引转换为实际位置
            cycle_positions = [positions[i] for i in repeats]
            return cycle_positions
    
    # 如果没有找到明显的重复模式，尝试基于主要聚类的分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) > 1:
        # 寻找最常见的聚类
        dominant_label = unique_labels[np.argmax(counts)]
        dominant_positions = positions[labels == dominant_label]
        
        if len(dominant_positions) > 1:
            # 以主要聚类的位置间隔作为周期
            intervals = np.diff(dominant_positions)
            if len(intervals) > 0:
                avg_interval = int(np.mean(intervals))
                cycle_starts = list(range(0, positions[-1], avg_interval))
                return cycle_starts
    
    return [0]

def analyze_cluster_intervals(positions, labels):
    """
    分析每个聚类内部段之间的时间间隔
    """
    cluster_intervals = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        cluster_positions = positions[labels == label]
        if len(cluster_positions) > 1:
            # 排序位置
            cluster_positions_sorted = np.sort(cluster_positions)
            # 计算间隔
            intervals = np.diff(cluster_positions_sorted)
            cluster_intervals[label] = {
                'positions': cluster_positions_sorted,
                'intervals': intervals,
                'min_interval': np.min(intervals) if len(intervals) > 0 else 0,
                'mean_interval': np.mean(intervals) if len(intervals) > 0 else 0,
                'std_interval': np.std(intervals) if len(intervals) > 0 else 0,
                'count': len(cluster_positions)
            }
        else:
            cluster_intervals[label] = {
                'positions': positions[labels == label],
                'intervals': [],
                'min_interval': 0,
                'mean_interval': 0,
                'std_interval': 0,
                'count': len(positions[labels == label])
            }
    
    return cluster_intervals

def find_periodic_clusters(cluster_intervals, tolerance=0.15):
    """
    基于最小间隔找到具有相似周期的聚类
    tolerance: 相似度容忍度，默认15%
    """
    if not cluster_intervals:
        return []
    
    # 提取有效的最小间隔（排除间隔为0或过小的）
    valid_intervals = {}
    for label, info in cluster_intervals.items():
        if info['min_interval'] > 0 and info['count'] >= 2:  # 至少有2个段才能计算间隔
            valid_intervals[label] = info['min_interval']
    
    if len(valid_intervals) < 2:
        return []
    
    # 找到相似的最小间隔
    periodic_groups = []
    processed_labels = set()
    
    for label1, interval1 in valid_intervals.items():
        if label1 in processed_labels:
            continue
            
        similar_group = [label1]
        processed_labels.add(label1)
        
        for label2, interval2 in valid_intervals.items():
            if label2 in processed_labels:
                continue
                
            # 计算相对差异
            relative_diff = abs(interval1 - interval2) / max(interval1, interval2)
            if relative_diff <= tolerance:
                similar_group.append(label2)
                processed_labels.add(label2)
        
        if len(similar_group) >= 2:  # 至少有2个聚类具有相似间隔
            avg_interval = np.mean([valid_intervals[label] for label in similar_group])
            periodic_groups.append({
                'labels': similar_group,
                'avg_interval': avg_interval,
                'intervals': [valid_intervals[label] for label in similar_group]
            })
    
    # 按平均间隔排序，选择最可能的周期
    periodic_groups.sort(key=lambda x: len(x['labels']), reverse=True)
    
    return periodic_groups

def construct_main_loop_from_periodic_clusters(positions, labels, periodic_groups, cluster_intervals):
    """
    从周期性聚类中构建主循环区域
    """
    if not periodic_groups:
        return None
    
    # 选择最有可能的周期组（包含聚类最多的组）
    best_group = periodic_groups[0]
    periodic_labels = best_group['labels']
    estimated_period = best_group['avg_interval']
    
    print(f"检测到周期性聚类: {periodic_labels}")
    print(f"估计周期长度: {estimated_period:.1f}")
    
    # 收集所有周期性聚类的位置
    all_periodic_positions = []
    for label in periodic_labels:
        cluster_pos = cluster_intervals[label]['positions']
        all_periodic_positions.extend(cluster_pos)
    
    all_periodic_positions = np.sort(all_periodic_positions)
    
    if len(all_periodic_positions) < 3:
        return None
    
    # 寻找连续的周期性区域
    # 基于估计的周期长度，找到最长的连续周期性区域
    continuous_regions = []
    current_region_start = all_periodic_positions[0]
    current_region_positions = [all_periodic_positions[0]]
    
    for i in range(1, len(all_periodic_positions)):
        current_pos = all_periodic_positions[i]
        prev_pos = all_periodic_positions[i-1]
        
        # 如果间隔接近估计的周期长度，认为是连续的
        gap = current_pos - prev_pos
        if gap <= estimated_period * 1.5:  # 允许1.5倍的容忍度
            current_region_positions.append(current_pos)
        else:
            # 当前区域结束，保存并开始新区域
            if len(current_region_positions) >= 3:  # 至少3个位置才算有效区域
                continuous_regions.append({
                    'start': current_region_start,
                    'end': current_region_positions[-1],
                    'positions': current_region_positions.copy(),
                    'length': len(current_region_positions)
                })
            
            # 开始新区域
            current_region_start = current_pos
            current_region_positions = [current_pos]
    
    # 处理最后一个区域
    if len(current_region_positions) >= 3:
        continuous_regions.append({
            'start': current_region_start,
            'end': current_region_positions[-1],
            'positions': current_region_positions.copy(),
            'length': len(current_region_positions)
        })
    
    if not continuous_regions:
        return None
    
    # 选择最长的连续区域作为主循环
    main_loop_region = max(continuous_regions, key=lambda x: x['length'])
    
    return {
        'start': main_loop_region['start'],
        'end': main_loop_region['end'],
        'periodic_labels': periodic_labels,
        'estimated_period': estimated_period,
        'cycle_positions': main_loop_region['positions'],
        'all_regions': continuous_regions
    }

def detect_three_phases(positions, labels, signal_1d):
    """
    检测周期性运动的三个阶段：准备期、主循环期、跳出期
    基于周期性聚类间隔分析的改进版本
    """
    if len(labels) < 6:
        return None, None, None
    
    # 分析聚类间隔
    cluster_intervals = analyze_cluster_intervals(positions, labels)
    
    print("\n=== 聚类间隔分析 ===")
    for label, info in cluster_intervals.items():
        print(f"聚类 {label}: 段数={info['count']}, 最小间隔={info['min_interval']:.1f}, "
              f"平均间隔={info['mean_interval']:.1f}±{info['std_interval']:.1f}")
    
    # 找到周期性聚类
    periodic_groups = find_periodic_clusters(cluster_intervals)
    
    if not periodic_groups:
        print("未检测到明显的周期性模式，使用传统方法...")
        # 回退到原来的基于稳定性的方法
        return detect_three_phases_traditional(positions, labels, signal_1d)
    
    print(f"\n检测到 {len(periodic_groups)} 个周期性组:")
    for i, group in enumerate(periodic_groups):
        print(f"组 {i+1}: 聚类 {group['labels']}, 平均间隔={group['avg_interval']:.1f}")
    
    # 构建主循环区域
    main_loop_info = construct_main_loop_from_periodic_clusters(
        positions, labels, periodic_groups, cluster_intervals
    )
    
    if not main_loop_info:
        print("无法构建主循环区域，使用传统方法...")
        return detect_three_phases_traditional(positions, labels, signal_1d)
    
    # 定义三个阶段
    main_start = main_loop_info['start']
    main_end = main_loop_info['end']
    total_length = len(signal_1d)
    
    # 准备期：从开始到主循环开始
    preparation_phase = (0, main_start) if main_start > 0 else None
    
    # 主循环期：周期性区域
    main_loop_phase = (main_start, main_end)
    
    # 跳出期：从主循环结束到数据结束
    exit_phase = (main_end, total_length) if main_end < total_length else None
    
    print(f"\n=== 基于周期性分析的三阶段划分 ===")
    if preparation_phase:
        print(f"准备期: {preparation_phase[0]} - {preparation_phase[1]} (长度: {preparation_phase[1] - preparation_phase[0]})")
    print(f"主循环期: {main_loop_phase[0]} - {main_loop_phase[1]} (长度: {main_loop_phase[1] - main_loop_phase[0]})")
    if exit_phase:
        print(f"跳出期: {exit_phase[0]} - {exit_phase[1]} (长度: {exit_phase[1] - exit_phase[0]})")
    
    return preparation_phase, main_loop_phase, exit_phase

def detect_three_phases_traditional(positions, labels, signal_1d):
    """
    传统的基于稳定性的三阶段检测方法（作为备用）
    """
    # 原来的实现逻辑
    label_stability = []
    window_size = max(3, len(labels) // 10)
    
    for i in range(len(labels)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(labels), i + window_size // 2 + 1)
        window_labels = labels[start_idx:end_idx]
        
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        max_consistency = np.max(counts) / len(window_labels)
        label_stability.append(max_consistency)
    
    label_stability = np.array(label_stability)
    
    # 计算信号变化率
    signal_variance = []
    for i in range(len(positions)):
        start_pos = positions[i]
        end_pos = start_pos + (positions[1] - positions[0]) if i < len(positions) - 1 else len(signal_1d)
        end_pos = min(end_pos, len(signal_1d))
        
        segment_signal = signal_1d[start_pos:end_pos]
        if len(segment_signal) > 1:
            variance = np.var(segment_signal)
        else:
            variance = 0
        signal_variance.append(variance)
    
    signal_variance = np.array(signal_variance)
    
    # 综合稳定性指标
    stability_score = label_stability * (1 / (1 + signal_variance / np.max(signal_variance)))
    
    # 使用滑动平均平滑稳定性分数
    smooth_window = max(3, len(stability_score) // 8)
    stability_smooth = np.convolve(stability_score, np.ones(smooth_window)/smooth_window, mode='same')
    
    # 寻找稳定区域
    stability_threshold = np.mean(stability_smooth) + 0.5 * np.std(stability_smooth)
    stable_regions = stability_smooth > stability_threshold
    
    if not np.any(stable_regions):
        total_length = len(positions)
        prep_end = total_length // 4
        exit_start = 3 * total_length // 4
        
        preparation_phase = (0, prep_end)
        main_loop_phase = (prep_end, exit_start)
        exit_phase = (exit_start, total_length)
    else:
        stable_segments = []
        start = None
        
        for i, is_stable in enumerate(stable_regions):
            if is_stable and start is None:
                start = i
            elif not is_stable and start is not None:
                stable_segments.append((start, i))
                start = None
        
        if start is not None:
            stable_segments.append((start, len(stable_regions)))
        
        if stable_segments:
            main_segment = max(stable_segments, key=lambda x: x[1] - x[0])
            main_start, main_end = main_segment
            
            preparation_phase = (0, main_start) if main_start > 0 else None
            main_loop_phase = (main_start, main_end)
            exit_phase = (main_end, len(positions)) if main_end < len(positions) else None
        else:
            total_length = len(positions)
            prep_end = total_length // 4
            exit_start = 3 * total_length // 4
            
            preparation_phase = (0, prep_end)
            main_loop_phase = (prep_end, exit_start)
            exit_phase = (exit_start, total_length)
    
    # 将段索引转换为时间位置
    def segment_to_position(segment_range):
        if segment_range is None:
            return None
        start_seg, end_seg = segment_range
        start_pos = positions[start_seg] if start_seg < len(positions) else positions[-1]
        end_pos = positions[end_seg-1] if end_seg <= len(positions) else len(signal_1d)
        return (start_pos, end_pos)
    
    prep_positions = segment_to_position(preparation_phase)
    main_positions = segment_to_position(main_loop_phase)
    exit_positions = segment_to_position(exit_phase)
    
    return prep_positions, main_positions, exit_positions

def plot_results(data, signal_1d, best_results, best_window_size, method, save_path=None):
    """绘制结果图"""
    if not best_results:
        print("没有有效结果可以绘制")
        return
    
    fig, axes = plt.subplots(7, 1, figsize=(15, 21))  # 增加一个子图用于阶段分析
    
    # 1. 原始多维数据
    axes[0].set_title('Original Multi-dimensional Data')
    for i in range(data.shape[1]):
        axes[0].plot(data[:, i], label=f'Dim {i+1}', alpha=0.7)
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. 降维后的一维信号和分段，增加三阶段标注
    axes[1].set_title(f'1D Signal with Segments and Three Phases (method: {method}, window: {best_window_size})')
    axes[1].plot(signal_1d, 'b-', linewidth=2)
    
    # 标记分段位置
    positions = best_results['positions']
    labels = best_results['labels']
    colors = plt.cm.tab10(np.linspace(0, 1, best_results['optimal_k']))
    
    for i, (pos, label) in enumerate(zip(positions, labels)):
        color = colors[label]
        axes[1].axvspan(pos, pos + best_window_size, alpha=0.3, color=color)
    
    # 标记三个阶段
    if 'preparation_phase' in best_results and best_results['preparation_phase']:
        prep_start, prep_end = best_results['preparation_phase']
        axes[1].axvspan(prep_start, prep_end, alpha=0.2, color='orange', 
                       label='Preparation Phase')
        axes[1].text(prep_start + (prep_end - prep_start)/2, 
                    np.max(signal_1d) * 0.9, 'Preparation\nPhase', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
    
    if 'main_loop_phase' in best_results and best_results['main_loop_phase']:
        main_start, main_end = best_results['main_loop_phase']
        axes[1].axvspan(main_start, main_end, alpha=0.2, color='green', 
                       label='Main Loop Phase')
        axes[1].text(main_start + (main_end - main_start)/2, 
                    np.max(signal_1d) * 0.8, 'Main Loop\nPhase', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7))
    
    if 'exit_phase' in best_results and best_results['exit_phase']:
        exit_start, exit_end = best_results['exit_phase']
        axes[1].axvspan(exit_start, exit_end, alpha=0.2, color='red', 
                       label='Exit Phase')
        axes[1].text(exit_start + (exit_end - exit_start)/2, 
                    np.max(signal_1d) * 0.9, 'Exit\nPhase', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
    
    # 标记周期起点
    if 'cycle_starts' in best_results:
        for start in best_results['cycle_starts']:
            axes[1].axvline(x=start, color='purple', linestyle='--', alpha=0.8, linewidth=2)
    
    axes[1].grid(True)
    axes[1].legend(loc='upper right')
    
    # 3. 聚类标签序列
    axes[2].set_title('Clustering Labels Sequence')
    axes[2].plot(positions, labels, 'o-', linewidth=2, markersize=6)
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Cluster Label')
    axes[2].grid(True)
    
    # 4. 每个聚类的段数分布
    cluster_counts = []
    cluster_ids = []
    for cluster_id, info in best_results['cluster_info'].items():
        cluster_ids.append(cluster_id)
        cluster_counts.append(info['count'])
    
    axes[3].set_title('Segments per Cluster')
    bars = axes[3].bar(cluster_ids, cluster_counts, alpha=0.7, color=colors[:len(cluster_ids)])
    axes[3].set_xlabel('Cluster ID')
    axes[3].set_ylabel('Number of Segments')
    axes[3].grid(True)
    
    # 5. 聚类质心可视化
    axes[4].set_title('Cluster Centroids')
    segments = best_results['segments']
    for cluster_id, info in best_results['cluster_info'].items():
        if info['count'] > 0:
            centroid = np.mean(info['segments'], axis=0)
            axes[4].plot(centroid, color=colors[cluster_id], linewidth=2, 
                        label=f'Cluster {cluster_id} (n={info["count"]})')
    axes[4].legend()
    axes[4].grid(True)
    
    # 6. DBSCAN结果对比
    axes[5].set_title('DBSCAN Clustering Results')
    dbscan_labels = best_results['dbscan_labels']
    unique_dbscan_labels = np.unique(dbscan_labels)
    dbscan_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_dbscan_labels)))
    
    for i, (pos, label) in enumerate(zip(positions, dbscan_labels)):
        if label != -1:  # 排除噪声点
            color_idx = np.where(unique_dbscan_labels == label)[0][0]
            color = dbscan_colors[color_idx]
        else:
            color = 'black'  # 噪声点用黑色
        axes[5].scatter(pos, label, color=color, s=50, alpha=0.7)
    
    axes[5].set_xlabel('Position')
    axes[5].set_ylabel('DBSCAN Label')
    axes[5].grid(True)
    
    # 7. 新增：三阶段分析图和间隔分析
    axes[6].set_title('Three-Phase Analysis with Interval Analysis')
    
    # 计算并显示每个阶段的特征
    phase_info = []
    phase_colors = ['orange', 'green', 'red']
    phase_names = ['Preparation', 'Main Loop', 'Exit']
    phases = [best_results.get('preparation_phase'), 
              best_results.get('main_loop_phase'), 
              best_results.get('exit_phase')]
    
    for i, (phase, color, name) in enumerate(zip(phases, phase_colors, phase_names)):
        if phase:
            start_pos, end_pos = phase
            phase_signal = signal_1d[start_pos:end_pos]
            
            # 计算阶段特征
            phase_mean = np.mean(phase_signal)
            phase_std = np.std(phase_signal)
            phase_length = end_pos - start_pos
            
            # 在图上显示阶段区域
            axes[6].axvspan(start_pos, end_pos, alpha=0.3, color=color, label=f'{name} Phase')
            
            # 添加文本标注
            axes[6].text(start_pos + phase_length/2, phase_mean,
                        f'{name}\nLen: {phase_length}\nStd: {phase_std:.3f}',
                        ha='center', va='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
            
            phase_info.append({
                'name': name,
                'start': start_pos,
                'end': end_pos,
                'length': phase_length,
                'mean': phase_mean,
                'std': phase_std
            })
    
    # 显示间隔分析结果
    cluster_intervals = analyze_cluster_intervals(positions, labels)
    interval_text = "Cluster Intervals:\n"
    for label, info in cluster_intervals.items():
        if info['min_interval'] > 0:
            interval_text += f"C{label}: {info['min_interval']:.1f}±{info['std_interval']:.1f} "
    
    axes[6].text(0.02, 0.98, interval_text, transform=axes[6].transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    axes[6].plot(signal_1d, 'b-', linewidth=1, alpha=0.7)
    axes[6].set_xlabel('Time Steps')
    axes[6].set_ylabel('Signal Amplitude')
    axes[6].legend()
    axes[6].grid(True)
    
    plt.tight_layout()
    # 先新建目录
    # save_path 形如 /data1/haoran/code/LoopDetector/methods/clustering_analysis/{task_name}/{task_cfg}/results/{episode_id}.png
    if save_path:
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出阶段统计信息
    print(f"\n=== 三阶段分析结果 ===")
    for info in phase_info:
        print(f"{info['name']} Phase:")
        print(f"  时间范围: {info['start']} - {info['end']}")
        print(f"  长度: {info['length']} steps")
        print(f"  平均值: {info['mean']:.4f}")
        print(f"  标准差: {info['std']:.4f}")
        print(f"  变异系数: {info['std']/abs(info['mean']):.4f}" if info['mean'] != 0 else "  变异系数: N/A")
    
    # 输出间隔分析结果
    print(f"\n=== 聚类间隔分析结果 ===")
    for label, info in cluster_intervals.items():
        print(f"聚类 {label}:")
        print(f"  段数: {info['count']}")
        if info['min_interval'] > 0:
            print(f"  最小间隔: {info['min_interval']:.1f}")
            print(f"  平均间隔: {info['mean_interval']:.1f}")
            print(f"  间隔标准差: {info['std_interval']:.1f}")
            print(f"  间隔变异系数: {info['std_interval']/info['mean_interval']:.3f}" if info['mean_interval'] > 0 else "  间隔变异系数: N/A")

def main():

    # ========= fill below =========

    # 加载数据
    task_name = "shake_bottle_loop"
    task_cfg = "loop3"
    episode_id = 0

    # ========= fill above =========

    data = np.load(f'/home/liaohaoran/code/RoboTwin/data/utils/data_for_analyse/{task_name}/{task_cfg}/episode{episode_id}_joint_action.npy')[:, :7]
    print(f"数据形状: {data.shape}")
    # 只使用sum方法
    method = 'sum'
    results = {}
    
    print(f"\n{'='*50}")
    print(f"方法: {method}")
    print('='*50)
    
    signal_1d, best_results, best_window_size, all_results = clustering_analysis(data, method)
    
    results[method] = {
        'signal_1d': signal_1d,
        'best_results': best_results,
        'best_window_size': best_window_size,
        'all_results': all_results
    }
    
    # 绘制结果
    plot_results(data, signal_1d, best_results, best_window_size, method, save_path=f'/home/liaohaoran/code/RoboTwin/data/utils/data_for_analyse/{task_name}/{task_cfg}/results/{episode_id}.png')
    
    if best_results:
        print(f"最佳窗口大小: {best_window_size}")
        print(f"轮廓系数: {best_results['silhouette_score']:.4f}")
        print(f"聚类数: {best_results['optimal_k']}")
        if 'cycle_starts' in best_results:
            print(f"检测到 {len(best_results['cycle_starts'])} 个周期")
            print(f"周期起点: {best_results['cycle_starts']}")
            print(f"平均周期长度: {best_results['avg_period']}")
        
        # 输出三阶段信息
        if 'preparation_phase' in best_results:
            prep = best_results['preparation_phase']
            main = best_results['main_loop_phase']
            exit_phase = best_results['exit_phase']
            
            print(f"\n--- 三阶段分析 ---")
            if prep:
                print(f"准备期: {prep[0]}-{prep[1]} (长度: {prep[1]-prep[0]})")
            if main:
                print(f"主循环期: {main[0]}-{main[1]} (长度: {main[1]-main[0]})")
            if exit_phase:
                print(f"跳出期: {exit_phase[0]}-{exit_phase[1]} (长度: {exit_phase[1]-exit_phase[0]})")
            
            # 计算各阶段占比
            total_length = len(signal_1d)
            if prep:
                prep_ratio = (prep[1] - prep[0]) / total_length * 100
                print(f"准备期占比: {prep_ratio:.1f}%")
            if main:
                main_ratio = (main[1] - main[0]) / total_length * 100
                print(f"主循环期占比: {main_ratio:.1f}%")
            if exit_phase:
                exit_ratio = (exit_phase[1] - exit_phase[0]) / total_length * 100
                print(f"跳出期占比: {exit_ratio:.1f}%")
    
    # 保存结果
    # np.save('/data1/haoran/code/LoopDetector/methods/clustering_analysis/results.npy', results)
    # print(f"\n结果已保存到 methods/clustering_analysis/results.npy")

if __name__ == "__main__":
    main()

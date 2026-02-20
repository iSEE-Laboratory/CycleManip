import numpy as np
from typing import List

def count_cycles(sequence: List[float]) -> int:
    """
    统计一维序列中的周期数
    
    参数:
        sequence: 一维序列数据
        
    返回:
        统计得到的周期数
    """
    if len(sequence) < 2:
        return 0
    
    # 1. 数据归一化到0-1
    min_val = min(sequence)
    max_val = max(sequence)
    
    # 如果序列是常数，没有周期
    if max_val == min_val:
        return 0
    
    normalized_seq = [(x - min_val) / (max_val - min_val) for x in sequence]
    
    # 2. 线性插值使数据更连续
    # 在每两个数据点之间插入9个点，总共10倍的数据点
    interpolated_seq = []
    for i in range(len(normalized_seq) - 1):
        interpolated_seq.append(normalized_seq[i])
        # 在两个点之间插入9个点
        for j in range(1, 10):
            interp_val = normalized_seq[i] + (normalized_seq[i+1] - normalized_seq[i]) * j / 10
            interpolated_seq.append(interp_val)
    interpolated_seq.append(normalized_seq[-1])
    
    # 3. 对0-1范围进行100次切片
    threshold_levels = np.linspace(0, 1, 100)
    cycle_counts = []
    
    for threshold in threshold_levels:
        # 统计序列中等于当前阈值的位置
        equal_points = []
        
        # 由于浮点数精度问题，使用小的容差来判断相等
        tolerance = 5e-4
        for i, val in enumerate(interpolated_seq):
            if abs(val - threshold) < tolerance:
                equal_points.append(i)
        
        # 如果找到的相等点数量是奇数，排除
        if len(equal_points) % 2 != 0:
            continue
        
        # 如果是偶数，计算周期数
        if len(equal_points) >= 2:
            cycle_count = len(equal_points) // 2
            cycle_counts.append(cycle_count)
    
    # 4. 找到出现次数最多的周期数
    if not cycle_counts:
        return 0
    
    # 统计每个周期数出现的频率
    from collections import Counter
    count_freq = Counter(cycle_counts)
    
    # 返回出现频率最高的周期数
    most_common = count_freq.most_common(1)
    return most_common[0][0]

# 测试函数
def test_count_cycles():
    """测试函数"""
    # 创建一个简单的周期序列进行测试
    t = np.linspace(0, 4*np.pi, 400)
    test_sequence = np.sin(t).tolist()  # 2个完整周期
    
    result = count_cycles(test_sequence)
    print(f"测试序列的周期数: {result}")
    
    # 创建更多周期的序列测试
    t2 = np.linspace(0, 10*np.pi, 400)
    test_sequence2 = np.sin(t2).tolist()  # 5个完整周期
    
    result2 = count_cycles(test_sequence2)
    print(f"更多周期测试序列的周期数: {result2}")

if __name__ == "__main__":
    test_count_cycles()
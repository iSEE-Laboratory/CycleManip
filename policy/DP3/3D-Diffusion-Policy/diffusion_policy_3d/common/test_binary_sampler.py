import numpy as np
import time

def binary_sample_indices_long(start_idx: int, end_idx: int, n_samples: int) -> np.ndarray:
    """
    偏右边界的二分采样索引
    """
    indices = np.zeros(n_samples, dtype=np.int64)
    assert n_samples > 0, "n_samples must be positive"

    left = mid = start_idx
    right = end_idx - 1
    
    indices[0] = start_idx  # 确保第一个索引是起始索引
    cnt = 1
    
    while cnt < n_samples:
        # 当前区间长度
        section_len = right - left + 1
        # 如果当前区间长度小于等于剩余采样数，则直接将剩余的采样点填充为右边界
        remain = n_samples - cnt
        if section_len <= remain:
            indices[cnt:cnt + section_len] = np.arange(left, right + 1)
            cnt += section_len
            while cnt < n_samples:
                indices[cnt] = right
                cnt += 1
            
            break

        mid = left + (right - left) // 2
        indices[cnt] = mid
        cnt += 1

        left = mid + 1
        
    return indices

def binary_sample_indices_short(start_idx: int, end_idx: int, n_samples: int) -> np.ndarray:
    """
    从右到左的二分采样法
    从当前时刻往回数取 n_samples 个索引，
    类似 ...64 32 16 8 4 2 1 now 的指数级采样
    """
    length = end_idx - start_idx

    # 1. 生成 2^i - 1 序列（不超过 length）
    # 和原版一致：从右往左的指数偏移
    powers = 2 ** np.arange(n_samples)
    powers = powers[powers < length]
    exp_seq = powers - 1
    
    # print(f"Exponential sequence before trimming: {exp_seq}")

    # 2. 如果不够 n_samples，则补充剩余的最小索引
    if len(exp_seq) < n_samples:
        all_indices = np.arange(length)
        mask = np.ones(length, dtype=np.bool_)
        mask[exp_seq] = False
        extra = all_indices[mask][:n_samples - len(exp_seq)]
        exp_seq = np.concatenate((exp_seq, extra))
        
    if len(exp_seq) < n_samples:
        # 如果仍然不够，补0
        exp_seq = np.concatenate([exp_seq, np.zeros(n_samples - len(exp_seq), dtype=np.int64)])
        
    # print(f"Exponential sequence after trimming: {exp_seq}")

    # 3. 反向映射到全局索引
    indices = end_idx - 1 - exp_seq
    indices.sort()

    return indices



def binary_sample_indices_for(start_idx: int, end_idx: int, n_samples: int) -> np.ndarray:
    """
    从右到左的二分采样法
    从当前时刻往回数取n_samples个索引，
    ...64 32 16 8 4 2 1 now 等指数级采样
    """
    indices = np.zeros(n_samples, dtype=np.int64)
    
    # 先判断区间是否够长
    length = end_idx - start_idx
    
    # 构造出倒序采样数组
    """
    1. 获取倒叙采样索引
    """
    _indices = [2 ** i - 1 for i in range(0, n_samples) if 2 ** i < length]
    # len中除去_indices中的，按照从小到大排序 用np的setdiff1d获取差集
    available = np.setdiff1d(np.arange(start_idx, end_idx), _indices)
    # available = np.sort(available)
    
    while len(_indices) < n_samples:
        # 如果_indices长度小于n_samples，继续添加
        if len(available) > 0:
            _indices.append(available[0])
            available = available[1:]
        else:
            # 如果没有可用的索引了，填充0
            _indices.append(0)
    
    def insertion_sort_optimized(arr):
        """
        对于短序列，且局部有序，插入排序的时间复杂度可以接近O(n)。
        """
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            
            # 如果前一个元素已经 <= key，无需移动
            if arr[j] <= key:
                continue
            
            # 否则执行插入
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
    
    _indices = insertion_sort_optimized(_indices)
    
    """转化成index数组"""
    
    indices = end_idx - 1 - np.array(_indices, dtype=np.int64)
    
    indices = indices[::-1]  # 反转，使其从小到大排列
    
    # print(f"Binary sample indices: {indices}")
        
    return indices


def binary_sample_indices_short(start_idx: int, end_idx: int, n_samples: int) -> np.ndarray:
    """
    从右到左的二分采样法
    从当前时刻往回数取 n_samples 个索引，
    类似 ...64 32 16 8 4 2 1 now 的指数级采样
    """
    length = end_idx - start_idx

    # 1. 生成 2^i - 1 序列（不超过 length）
    # 和原版一致：从右往左的指数偏移
    powers = 2 ** np.arange(n_samples)
    powers = powers[powers < length]
    exp_seq = powers - 1
    
    # print(f"Exponential sequence before trimming: {exp_seq}")

    # 2. 如果不够 n_samples，则补充剩余的最小索引
    if len(exp_seq) < n_samples:
        all_indices = np.arange(length)
        mask = np.ones(length, dtype=np.bool_)
        mask[exp_seq] = False
        extra = all_indices[mask][:n_samples - len(exp_seq)]
        exp_seq = np.concatenate((exp_seq, extra))
        
    if len(exp_seq) < n_samples:
        # 如果仍然不够，补0
        exp_seq = np.concatenate([exp_seq, np.zeros(n_samples - len(exp_seq), dtype=np.int64)])
        
    # print(f"Exponential sequence after trimming: {exp_seq}")

    # 3. 反向映射到全局索引
    indices = end_idx - 1 - exp_seq
    indices.sort()

    return indices



def long_short_sample_indices(start_idx: int, end_idx: int, n_samples: int) -> np.ndarray:
    
    def binary_sample_indices_long(start_idx: int, end_idx: int, n_samples: int) -> np.ndarray:
        """
        偏右边界的二分采样索引
        """
        indices = np.zeros(n_samples, dtype=np.int64)
        assert n_samples > 0, "n_samples must be positive"

        left = mid = start_idx
        right = end_idx - 1
        
        indices[0] = start_idx  # 确保第一个索引是起始索引
        cnt = 1
        
        while cnt < n_samples:
            # 当前区间长度
            section_len = right - left + 1
            # 如果当前区间长度小于等于剩余采样数，则直接将剩余的采样点填充为右边界
            remain = n_samples - cnt
            if section_len <= remain:
                indices[cnt:cnt + section_len] = np.arange(left, right + 1)
                cnt += section_len
                while cnt < n_samples:
                    indices[cnt] = right
                    cnt += 1
                
                break

            mid = left + (right - left) // 2
            indices[cnt] = mid
            cnt += 1

            left = mid + 1
            
        return indices

    def binary_sample_indices_short(start_idx: int, end_idx: int, n_samples: int) -> np.ndarray:
        """
        从右到左的二分采样法
        从当前时刻往回数取 n_samples 个索引，
        类似 ...64 32 16 8 4 2 1 now 的指数级采样
        """
        length = end_idx - start_idx

        # 1. 生成 2^i - 1 序列（不超过 length）
        # 和原版一致：从右往左的指数偏移
        powers = 2 ** np.arange(n_samples)
        powers = powers[powers < length]
        exp_seq = powers - 1
        
        # print(f"Exponential sequence before trimming: {exp_seq}")

        # 2. 如果不够 n_samples，则补充剩余的最小索引
        if len(exp_seq) < n_samples:
            all_indices = np.arange(length)
            mask = np.ones(length, dtype=np.bool_)
            mask[exp_seq] = False
            extra = all_indices[mask][:n_samples - len(exp_seq)]
            exp_seq = np.concatenate((exp_seq, extra))
            
        if len(exp_seq) < n_samples:
            # 如果仍然不够，补0
            exp_seq = np.concatenate((exp_seq, np.zeros(n_samples - len(exp_seq), dtype=np.int64)))
            
        # print(f"Exponential sequence after trimming: {exp_seq}")

        # 3. 反向映射到全局索引
        indices = end_idx - 1 - exp_seq
        indices.sort()

        return indices

    n_long = n_samples // 2
    n_short = n_samples - n_long
    long_indices = binary_sample_indices_long(start_idx, end_idx, n_long)
    short_indices = binary_sample_indices_short(start_idx, end_idx, n_short)

    return np.concatenate((long_indices, short_indices))


if __name__ == "__main__":
    for i in range(1, 50):
        indices = long_short_sample_indices(0, i, 8)
        print("-"*30)
        print(f"Indices for range (0, {i}): \n{indices}")
        
        
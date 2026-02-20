from typing import Optional
import numpy as np
import numba
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,  # 每个episode的结束索引
    sequence_length: int,      # 序列长度
    episode_mask: np.ndarray,  # episode掩码，用于标记哪些episode可用
    pad_before: int = 0,       # 序列开始前的填充长度
    pad_after: int = 0,        # 序列结束后的填充长度
    debug: bool = True,        # 是否开启调试模式
) -> np.ndarray:
    """
    创建采样索引，用于从replay buffer中提取序列数据
    
    返回格式: [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
    - buffer_start_idx: 在buffer中的起始索引
    - buffer_end_idx: 在buffer中的结束索引  
    - sample_start_idx: 在采样序列中的起始位置
    - sample_end_idx: 在采样序列中的结束位置
    """
    episode_mask.shape == episode_ends.shape
    # 限制填充长度在合理范围内
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # 跳过被掩码的episode
            continue
        
        # 计算当前episode的起始和结束索引
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        # 计算可能的序列起始位置范围
        # 允许序列开始位置为负数（需要前填充）
        min_start = -pad_before
        # 允许序列超出episode结束（需要后填充）
        max_start = episode_length - sequence_length + pad_after

        # 为每个可能的起始位置创建索引
        for idx in range(min_start, max_start + 1):
            # 计算在buffer中实际的起始和结束位置
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            
            # 计算需要填充的偏移量
            start_offset = buffer_start_idx - (idx + start_idx)  # 前填充量
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx  # 后填充量
            
            # 计算在采样序列中的有效数据位置
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            
            if debug:
                # 调试断言，确保计算正确
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    """
    生成验证集掩码
    
    Args:
        n_episodes: 总episode数量
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        val_mask: 布尔数组，True表示该episode用于验证
    """
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # 确保至少有1个episode用于验证，至少1个episode用于训练
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    """
    对掩码进行下采样，限制数据量
    
    Args:
        mask: 原始掩码
        max_n: 最大保留的数据量
        seed: 随机种子
    
    Returns:
        train_mask: 下采样后的掩码
    """
    train_mask = mask
    # 如果当前数据量超过最大限制，进行随机下采样
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]  # 获取当前为True的索引
        rng = np.random.default_rng(seed=seed)
        # 随机选择要保留的索引
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        
        # 创建新的掩码
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


class SequenceSampler:
    """
    序列采样器类，用于从replay buffer中采样固定长度的序列数据
    """

    def __init__(
            self,
            replay_buffer: ReplayBuffer,        # replay buffer对象
            sequence_length: int,               # 序列长度
            pad_before: int = 0,                # 前填充长度
            pad_after: int = 0,                 # 后填充长度
            keys=None,                          # 要采样的数据键名列表
            key_first_k=dict(),                 # 性能优化：只取某些键的前k个数据
            episode_mask: Optional[np.ndarray] = None,  # episode掩码
    ):
        """
        初始化序列采样器
        
        Args:
            key_first_k: dict str: int
                性能优化参数，只从指定键中取前k个数据（提高性能）
        """

        super().__init__()
        assert sequence_length >= 1
        
        # 如果没有指定键名，使用replay buffer中的所有键
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        # 如果没有提供episode掩码，默认所有episode都可用
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        # 如果有可用的episode，创建采样索引
        if np.any(episode_mask):
            indices = create_indices(
                episode_ends,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
            )
        else:
            # 如果没有可用episode，创建空索引数组
            indices = np.zeros((0, 4), dtype=np.int64)

        # 存储采样索引: (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices
        self.keys = list(keys)  # 防止OmegaConf列表性能问题
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

    def __len__(self):
        """返回可采样的序列数量"""
        return len(self.indices)

    def sample_sequence(self, idx):
        """
        根据索引采样一个序列
        
        Args:
            idx: 序列索引
            
        Returns:
            result: 包含各个键数据的字典
        """
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (self.indices[idx])
        result = dict()
        
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            
            # 性能优化：如果可能的话避免小内存分配
            if key not in self.key_first_k:
                # 正常情况：取完整的数据段
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # 性能优化：只加载需要使用的观测步骤
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                
                # 用NaN填充未加载的区域以便捕获bug
                # 未加载的区域不应该被使用
                sample = np.full(
                    (n_data, ) + input_arr.shape[1:],
                    fill_value=np.nan,
                    dtype=input_arr.dtype,
                )
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx + k_data]
                except Exception as e:
                    import pdb
                    pdb.set_trace()
            
            data = sample
            # 如果需要填充（序列长度不足或需要padding）
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                # 创建完整长度的数据数组
                data = np.zeros(
                    shape=(self.sequence_length, ) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                # 前填充：用第一个元素填充
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                # 后填充：用最后一个元素填充
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                # 填入实际数据
                data[sample_start_idx:sample_end_idx] = sample
            
            result[key] = data
        return result

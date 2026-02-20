import numpy as np
import numba
from typing import Optional
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler_strategy import past_k_sample_indices, past_k_sample_step_indices, adaptive_joint_sample_indices, long_short_sample_indices, adaptive_fixthre_endpose_sample_indices, next_k_sample_indices

@numba.jit(nopython=True)
def create_binary_indices(
    episode_ends: np.ndarray,
    action_horizon: int,
    obs_horizon: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    debug: bool = True,
) -> np.ndarray:
    """
    创建二分采样索引
    
    这个函数为每个episode中的每个有效时间步创建采样索引。
    对于每个时间步，定义了两个重要的时间窗口：
    1. 动作窗口：从当前时刻开始的未来action_horizon步动作
    2. 观测窗口：从episode开始到当前时刻的所有观测（后续会进行二分采样）
    
    Args:
        episode_ends: [num_episodes,] 每个episode的结束位置（全局索引）
        action_horizon: 动作预测的时间长度，即需要预测多少步未来动作
        obs_horizon: 观测的时间长度，即需要采样多少步历史观测  
        episode_mask: [num_episodes,] bool数组，标记哪些episode参与采样
        pad_before: 动作序列开始前的填充步数（暂未使用）
        pad_after: 动作序列结束后的填充步数，允许在episode末尾进行一定的填充
        debug: 调试标志（暂未使用）
    
    Returns:
        indices: [N, 6] 数组，每行包含一个有效采样点的信息：
        [action_start, action_end, obs_start, obs_end, episode_start, current_step]
        - action_start: 动作序列的起始全局索引
        - action_end: 动作序列的结束全局索引（不包含）
        - obs_start: 观测序列的起始全局索引（episode开始）
        - obs_end: 观测序列的结束全局索引（当前时刻+1，不包含）
        - episode_start: 当前episode的起始全局索引
        - current_step: 当前时刻在episode内的相对位置
    """
    # 验证输入参数的形状匹配
    episode_mask.shape == episode_ends.shape
    
    # 限制填充参数的范围，确保不超过action_horizon
    pad_before = min(max(pad_before, 0), action_horizon - 1)
    pad_after = min(max(pad_after, 0), action_horizon - 1)

    # 存储所有有效的采样索引
    indices = []
    
    max_length = 0
    # 遍历每个episode
    for i in range(len(episode_ends)):
        # 跳过被mask掉的episode
        if not episode_mask[i]:
            continue
            
        # 计算当前episode的起始和结束位置
        start_idx = 0  # 第一个episode从0开始
        if i > 0:
            start_idx = episode_ends[i - 1]  # 后续episode从前一个episode结束位置开始
        end_idx = episode_ends[i]  # 当前episode的结束位置
        episode_length = end_idx - start_idx  # 当前episode的长度

        # 确保episode长度足够提供obs_horizon个观测
        # 如果episode太短，无法提供足够的历史观测，跳过整个episode
        if episode_length < obs_horizon:
            continue

        if episode_length > max_length:
            max_length = episode_length

        # 计算有效的当前时刻范围
        # min_current_step: 当前时刻的最小值，必须保证有obs_horizon个历史观测
        # 由于obs_horizon包括当前时刻，所以最小值是obs_horizon-1
        # min_current_step = obs_horizon - 1
        
        # 这里将 min_current_step 设为 0，是因为在实际部署时，第一帧只能从 episode 的起始帧开始推断后续动作。
        # 这样设置可以保证训练集也覆盖到这些边界情况，提高模型的初步动作效果
        min_current_step = 0
        
        # max_current_step: 当前时刻的最大值，必须保证有action_horizon个未来动作
        # pad_after允许在episode末尾进行一定的填充
        max_current_step = episode_length - action_horizon + pad_after
        
        # 为episode中每个有效的时间步创建采样索引
        for current_step in range(min_current_step, max_current_step + 1):
            # 将episode内的相对位置转换为全局索引
            current_global_idx = start_idx + current_step
            
            # 定义动作序列的范围：从当前时刻开始的action_horizon步
            action_start = current_global_idx  # 动作序列起始位置（当前时刻）
            action_end = min(current_global_idx + action_horizon, end_idx)  # 动作序列结束位置
            
            # 检查动作序列是否足够长
            # 如果不够action_horizon步，跳过这个时间点
            if action_end - action_start < action_horizon:
                continue
            
            # 定义观测序列的范围：从episode开始到当前时刻（包含当前时刻）
            obs_episode_start = start_idx  # 观测序列起始位置（episode开始）
            obs_current_end = current_global_idx + 1  # 观测序列结束位置（当前时刻+1，不包含）
            
            # 添加采样索引到结果列表
            # 格式：[action_start, action_end, obs_start, obs_end, episode_start, current_step]
            indices.append([
                action_start, action_end,           # 动作序列的全局索引范围
                obs_episode_start, obs_current_end, # 观测序列的全局索引范围
                start_idx, end_idx, current_step, episode_length             # episode信息和当前步骤
            ])
            # current_step = 6, action_horizon=8, start_idx=0
            # 6, 14, 
            # 0, 7
            # 0, 6
    
    # 将列表转换为numpy数组并返回
    return np.array(indices, dtype=np.int64), max_length


class BinarySequenceSampler:
    """
    二分序列采样器
    - 动作：从当前时刻开始的future action_horizon步
    - 观测：从episode开始到当前时刻的二分采样obs_horizon步
    """
    
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        action_horizon: int,
        obs_horizon: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray] = None,
        sampler_strategy=None,
        action_sampling=None,
    ):
        super().__init__()
        assert action_horizon >= 1
        assert obs_horizon >= 1
        assert sampler_strategy != None

        self.sampler_strategy = sampler_strategy
        self.action_sampling = action_sampling
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices, max_length = create_binary_indices(
                episode_ends,
                action_horizon=action_horizon,
                obs_horizon=obs_horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
            )
        else:
            indices = np.zeros((0, 6), dtype=np.int64)


        self.max_length = max_length
        # [action_start, action_end, obs_start, obs_end, episode_start, current_step]
        self.indices = indices
        """
        例如
        000: array([168, 176, 161, 169, 161,   7])
        代表
        该episode的开始index是161
        当前时刻是7（相对于episode开始）
        需要加噪的动作序列是从168开始到176（不包含176）
        观测序列是从161开始到169（不包含169）（后续会二分采样）
        """
        self.keys = list(keys)
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        # 当前动作、结束时刻、观测开始时刻、观测结束时刻（index为当前时刻+1，在后续采样不会被采到）
        # 6, 14, 0, 7, 0, 6
        # action_data = [6:14]
        # state = [0:7]
        action_start, action_end, obs_start, obs_end, episode_start, episode_end, current_step, episode_length = self.indices[idx]
        result = dict()

        # history_state = self.replay_buffer['state']
        
        # state_diff = np.diff(self.replay_buffer['state'][obs_start: obs_end])
        # state_diff = np.insert(state_diff, 0, 0)

        if 'endpose' in self.replay_buffer:
            history_endpose = self.replay_buffer['endpose']
        #     ee_diff = np.diff(history_endpose[obs_start: obs_end])
        #     ee_diff = np.insert(ee_diff, 0, 0)
        # else:
        #     ee_diff = None

        short_state_indices = None
        
        if self.sampler_strategy == "past":
            input_indices = None
            pc_indices = past_k_sample_indices(obs_start, obs_end, self.obs_horizon)
            joint_indices = pc_indices

        elif self.sampler_strategy == "past_all":
            input_indices = None
            pc_indices = past_k_sample_indices(obs_start, obs_end, self.obs_horizon)
            joint_indices = range(obs_start, obs_end)

        elif self.sampler_strategy == "binary":
            input_indices = None
            pc_indices = long_short_sample_indices(obs_start, obs_end, self.obs_horizon)
            joint_indices = range(obs_start, obs_end)
            
        # elif self.sampler_strategy == "joint_adaptive-pc_adaptive_binary":
        #     input_indices = adaptive_joint_sample_indices(obs_start, obs_end, history_state)
        #     pc_indices = long_short_sample_indices(0, len(input_indices), self.obs_horizon)
        #     joint_indices = range(0, len(input_indices))
 
        # elif self.sampler_strategy == "joint_adaptive-pc_binary":
        #     input_indices = None
        #     joint_indices = adaptive_joint_sample_indices(obs_start, obs_end, history_state)
        #     pc_indices = long_short_sample_indices(obs_start, obs_end, self.obs_horizon)

        # elif self.sampler_strategy == "joint_adaptive-pc_past":
        #     input_indices = None
        #     joint_indices = adaptive_joint_sample_indices(obs_start, obs_end, history_state)
        #     pc_indices = past_k_sample_indices(obs_start, obs_end, self.obs_horizon)

        elif self.sampler_strategy == "joint_adaptive_fixthre_endpose-pc_adaptive_binary":
            input_indices = adaptive_fixthre_endpose_sample_indices(obs_start, obs_end, history_endpose)
            pc_indices = long_short_sample_indices(0, len(input_indices), self.obs_horizon)
            joint_indices = range(0, len(input_indices))

        elif self.sampler_strategy == "joint_adaptive_fixthre_endpose-pc_adaptive_past":
            input_indices = adaptive_fixthre_endpose_sample_indices(obs_start, obs_end, history_endpose)
            pc_indices = past_k_sample_indices(0, len(input_indices), self.obs_horizon)
            joint_indices = range(0, len(input_indices))

        elif self.sampler_strategy == "joint_all_pc_binary_short_state":
            input_indices = None
            pc_indices = long_short_sample_indices(obs_start, obs_end, self.obs_horizon)
            joint_indices = range(obs_start, obs_end)
            short_state_indices = past_k_sample_indices(obs_start, obs_end, self.obs_horizon)
        elif self.sampler_strategy == "adaptive_joint_all_pc_binary_short_state":
            input_indices = adaptive_fixthre_endpose_sample_indices(obs_start, obs_end, history_endpose)
            pc_indices = long_short_sample_indices(obs_start, obs_end, self.obs_horizon)
            joint_indices = range(obs_start, obs_end)
            short_state_indices = past_k_sample_indices(obs_start, obs_end, self.obs_horizon)
        elif self.sampler_strategy == "pc_uniform":
            input_indices = None
            pc_indices = np.linspace(obs_start, obs_end - 1, self.obs_horizon, dtype=np.int64)
            joint_indices = range(obs_start, obs_end)
        elif self.sampler_strategy == "pc_past_k_step":
            input_indices = None
            pc_indices = past_k_sample_step_indices(obs_start, obs_end, self.obs_horizon, 4)
            joint_indices = range(obs_start, obs_end)
        else:
            raise Exception(f"{self.sampler_strategy}")


        if self.action_sampling is not None:
            if self.action_sampling == "past_next":
                past_action_index = long_short_sample_indices(obs_start, obs_end, self.obs_horizon)
                next_action_index = next_k_sample_indices(action_start, episode_end, self.obs_horizon)
            else:
                raise Exception("1")
        else:
            past_action_index = None
            next_action_index = None

        for key in self.keys:

            input_arr = self.replay_buffer[key]

            if key == "action" :
                action_data = input_arr[action_start:action_end]
                if self.action_sampling is not None:
                    action_data = np.concatenate([action_data, input_arr[past_action_index], input_arr[next_action_index]])
                result[key] = {
                    'action': action_data,  # [action_horizon, ...]
                }

            elif "instruction" in key:
                result[key] = {
                    'obs': input_arr[obs_start],        # [...]
                }

            elif key in ["loop_counter", "loop_times"]:
                counter_data = input_arr[action_start]
                result[key] = {
                    'obs': np.array(counter_data),        # [...]
                }

            else:
                if input_indices is not None:
                    input_arr = input_arr[input_indices]
                if key in ['state', 'endpose']:
                    assert input_arr.ndim == 2, f"input_arr 必须是二维数组 (n, c)，但实际维度为 {input_arr.ndim}, {input_arr.shape}"
                    result[key] = {
                        'obs': input_arr[joint_indices],  # [..., ...]
                    }
                    assert input_arr[joint_indices].ndim == 2, f"input_arr[joint_indices] 必须是二维数组 (n, c)，但实际维度为 {input_arr[joint_indices].ndim}, {input_arr[joint_indices].shape}, joint_indices为 {joint_indices}"
                else:
                    result[key] = {
                        'obs': input_arr[pc_indices],     # [obs_horizon, ...]
                    } 

        # result["state_diff"] = {
        #         'obs': state_diff,        # [1]
        #     }
        # if ee_diff is not None:
        #     result["ee_diff"] = {
        #         'obs': ee_diff,        # [1]
        #     }
        if short_state_indices is not None:
            result["short_state"] = {
                'obs': self.replay_buffer["state"][short_state_indices]
            }
        if episode_length != None:
            result["loop_length"] = {
                'obs': np.array(episode_length),        # [1]
            }
            result["loop_curlen"] = {
                'obs': np.array(obs_end - obs_start),   # [1]
            }
            result["loop_now"] = {
                'obs': np.array((10*episode_length)//self.max_length),        # [1]
            }


        return result
    
    

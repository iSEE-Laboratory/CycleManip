import numpy as np
import numba
from scipy.spatial.transform import Rotation as R


def your_sampler(
        start_idx: int,
        end_idx: int,
        data: np.ndarray, # L, C
)  -> np.ndarray:
    # start_idx可以取到， end_idx不能被取到
    return # 返回全局索引 numpy array (sorted, unique) 例如 [start_idx, start_idx+7 , end_idx-1]


def adaptive_fixthre_endpose_sample_indices(
        start_idx: int,
        end_idx: int,
        traj: np.ndarray,
        rot_thresh: float = 10.0,       # deg
        trans_thresh: float = 0.015,    # m
        gripper_thresh: float = 0.5,
    ) -> np.ndarray:
    """
    按时间步遍历：对于每个时间 i，检查左右两侧（0/1）相对于它们各自上次被采样时刻
    的累计变化是否超过阈值。如果任一侧超过阈值，则采样 i（global idx = start_idx + i）。
    仅更新触发采样的侧的 last_sample。

    traj: shape (T, 14)  按顺序是 [L_xyz, L_rpy, L_gripper, R_xyz, R_rpy, R_gripper]
    返回全局索引 numpy array (sorted, unique)
    """
    assert 0 <= start_idx < end_idx <= traj.shape[0], f"invalid index, {start_idx}, {end_idx}"

    # 局部索引范围 [0, N)
    N = end_idx - start_idx
    if N <= 0:
        return np.array([], dtype=np.int64)

    # 以局部索引访问切片
    traj_slice = traj[start_idx:end_idx]

    # 初始化 last_sample（局部索引）
    last_sample = [0, 0]  # left, right
    key_indices = [0]     # 保留第一帧（局部索引0）

    # 便利的访问器：给 side 返回对应切片区间 offset
    def get_slice(i, side):
        off = 7 * side
        pos = traj_slice[i, off: off + 3]
        rpy = traj_slice[i, off + 3: off + 6]
        grip = float(traj_slice[i, off + 6])
        return pos, rpy, grip

    # 主循环：按时间步检查两侧
    for i in range(1, N):
        triggered = [False, False]

        for side in (0, 1):
            pos_i, rpy_i, grip_i = get_slice(i, side)
            pos_last, rpy_last, grip_last = get_slice(last_sample[side], side)

            # 平移差
            d_trans = np.linalg.norm(pos_i - pos_last)

            # 旋转差（scipy Rotation 单个对单个）
            r1 = R.from_euler('xyz', rpy_last)
            r2 = R.from_euler('xyz', rpy_i)
            d_rot = np.degrees((r2 * r1.inv()).magnitude())

            # gripper 差
            d_grip = abs(grip_i - grip_last)

            if (d_trans > trans_thresh) or (d_rot > rot_thresh) or (d_grip > gripper_thresh):
                triggered[side] = True

        # 如果任一侧触发，采样当前帧 i
        if triggered[0] or triggered[1]:
            key_indices.append(i)
            # 仅更新触发的侧的 last_sample（未触发的侧保留原 last_sample）
            for side in (0, 1):
                if triggered[side]:
                    last_sample[side] = i

    # 保证末帧被保留
    if key_indices[-1] != N - 1:
        key_indices.append(N - 1)

    # 转为全局索引并去重排序
    key_indices = np.array(key_indices, dtype=np.int64)
    global_idx = start_idx + np.unique(key_indices)

    return global_idx

def adaptive_endpose_sample_indices(    
        start_idx: int,
        end_idx: int,
        traj: np.ndarray,
        trans_thresh: float = 0.01,
        rot_thresh: float = 5,
        gripper_thresh: float = 0.3,
        ) -> np.ndarray:
    
    assert 0 <= start_idx < end_idx <= traj.shape[0], f"invalid index, {start_idx}, {end_idx}"

    def dis_trans_fun(x1, x2):
        return np.linalg.norm(x1 - x2)
    
    def dis_rot_fun(x1, x2):
        r1 = R.from_euler('xyz', x1)   # 弧度制
        r2 = R.from_euler('xyz', x2)
        r_rel = r2 * r1.inv()                        # 相对旋转
        dr_rad = r_rel.magnitude()                   # 最小旋转角（弧度）
        dr_deg = np.degrees(dr_rad)
        return dr_deg
    
    def dis_gripper_fun(x1, x2):
        return np.abs(x1 - x2)

    trans = traj[:, :3] # x, y, z
    rot = traj[:, 3:6] # roll, pitch, yaw
    gripper = traj[:, 6:] # 1

    dis_trans = dis_trans_fun(trans[1:], trans[:-1]) > trans_thresh
    dis_rot = dis_rot_fun(rot[1:], rot[:-1]) > rot_thresh
    dis_gripper = dis_gripper_fun(gripper[1:], gripper[:-1]) > gripper_thresh

    diffs = np.concatenate([dis_trans, dis_rot, dis_gripper], dim=-1)
    dists = diffs.max(axis=1)

    # === Step 2: 累计距离 & 分组 ===
    # 每当累计变化达到一个新的阈值倍数时，就采样一个关键帧
    cumsum = np.cumsum(dists)
    group_id = np.floor(cumsum).astype(np.int64)

    # === Step 3: 找到每个组第一次出现的位置 ===
    # 组号变化处即为新的采样点
    change_points = np.flatnonzero(np.diff(group_id, prepend=-1))  # prepend=-1 确保首帧被检测

    # 采样的全局索引
    idx = start_idx + change_points
    if len(idx) == 0 or idx[-1] != end_idx-1:
        idx = np.concatenate([idx, [end_idx-1]])
    return idx.astype(np.int64)

def backward_k_sample_indices(start_idx: int, end_idx: int, k: int) -> np.ndarray:
    """
    从start_idx到end_idx（不含end_idx）范围，从后往前每k帧抽取一帧
    
    参数:
        start_idx: 起始索引（包含）
        end_idx: 结束索引（不包含）
        k: 步长，每隔k帧抽取一次（k为正整数）
    
    返回:
        抽取的索引数组（从后往前排序，例如[9,7,5,...]）
    """
    # 处理无效范围（结束索引小于等于起始索引时返回空数组）
    if end_idx <= start_idx:
        return np.array([], dtype=np.int64)
    
    # 从后往前采样：起点是end_idx-1（范围内最后一个有效索引），
    # 终点是start_idx（确保包含start_idx），步长为-k（每k帧向前取一个）
    indices = np.arange(end_idx - 1, start_idx - 1, -k)
    
    return indices.astype(np.int64)

def past_k_sample_indices(start_idx: int, end_idx: int, n_samples: int) -> np.ndarray:
    # 正常情况下的索引范围：[end_idx - n_samples, ..., end_idx - 1]（不包含end_idx）
    indices = np.arange(end_idx - n_samples, end_idx)  # 终点改为end_idx，故最大为end_idx-1
    
    # 若范围起点小于 start_idx，说明不足 n_samples
    if indices[0] < start_idx:
        # 只取合法部分（[start_idx, end_idx - 1]）
        valid = np.arange(start_idx, end_idx)  # 终点为end_idx，故最大为end_idx-1
        pad_len = n_samples - len(valid)
        # 用 end_idx - 1 填充剩余位置（而非end_idx）
        indices = np.concatenate([valid, np.full(pad_len, end_idx - 1, dtype=np.int64)])
    
    return indices.astype(np.int64)

def past_k_sample_step_indices(start_idx: int, end_idx: int, n_samples: int, step: int = 1) -> np.ndarray:
    """
    从 [start_idx, end_idx-1] 范围内，从后往前以指定步长k采样n个索引
    Args:
        start_idx: 索引起始下限（包含）
        end_idx: 索引结束上限（不包含）
        n_samples: 需要采样的数量
        step: 跳步步长k（每隔k步取一个，默认1即连续采样）
    Returns:
        采样后的索引数组（长度固定为n_samples）
    """
    # 从后往前生成跳步索引：end_idx-1, end_idx-1-step, end_idx-1-2*step...
    indices = np.arange(end_idx - 1, end_idx - 1 - n_samples * step, -step)
    
    # 过滤掉小于start_idx的无效索引
    valid_indices = indices[indices >= start_idx]
    
    # 若有效索引不足n_samples，用最后一个有效值填充
    if len(valid_indices) < n_samples:
        pad_len = n_samples - len(valid_indices)
        # 若没有任何有效索引（极端情况），用start_idx填充
        pad_value = valid_indices[-1] if len(valid_indices) > 0 else start_idx
        indices = np.concatenate([valid_indices, np.full(pad_len, pad_value, dtype=np.int64)])
    else:
        indices = valid_indices
    
    # 保证数组长度为n_samples（防止步长导致超量），并按升序返回（和原函数逻辑一致）
    return indices[:n_samples].astype(np.int64)[::-1]

def adaptive_joint_sample_indices(
    start_idx: int,
    end_idx: int,
    traj: np.ndarray,
    dist_thresh: float = 0.1
) -> np.ndarray:
    """
    向量化版本：基于累计距离的自适应采样（自动决定采样数量）
    使用 np.cumsum + 分段检测，避免Python for循环。

    Args:
        start_idx (int): 起始索引（包含）
        end_idx (int): 结束索引（不包含）
        traj (list[np.ndarray]): 每帧的机械臂状态
        dist_thresh (float): 累积距离阈值，越小采样越密

    Returns:
        np.ndarray: 自动采样得到的帧索引，dtype=int64
    """
    # list[np.ndarray].shape 应该会报错 ——haoran
    assert 0 <= start_idx < end_idx <= traj.shape[0], f"invalid index, {start_idx}, {end_idx}"

    # === Step 1: 计算相邻帧间距离 ===
    eps = 1e-8  # 避免除零

    traj_np = traj[start_idx:end_idx]  # shape [L, C]
    min_vals = traj_np.min(axis=0, keepdims=True)  # (1, C)，每个通道的最小值
    max_vals = traj_np.max(axis=0, keepdims=True)  # (1, C)，每个通道的最大值
    traj_norm = (traj_np - min_vals) / (max_vals - min_vals + eps)  # (L, C)

    diffs = np.abs(traj_norm[1:] - traj_norm[:-1])  # (L-1, C) 取绝对值，防止负值
    dists = diffs.max(axis=1)  # (L-1,) 只要任何一个状态维度发生显著变化，就认为这是一个重要的变化时刻

    # === Step 2: 累计距离 & 分组 ===
    # 每当累计变化达到一个新的阈值倍数时，就采样一个关键帧
    cumsum = np.cumsum(dists)
    group_id = np.floor(cumsum / dist_thresh).astype(np.int64)

    # === Step 3: 找到每个组第一次出现的位置 ===
    # 组号变化处即为新的采样点
    change_points = np.flatnonzero(np.diff(group_id, prepend=-1))  # prepend=-1 确保首帧被检测

    # 采样的全局索引
    idx = start_idx + change_points
    if len(idx) == 0 or idx[-1] != end_idx-1:
        idx = np.concatenate([idx, [end_idx-1]])
    return idx.astype(np.int64)


@numba.jit(nopython=True)
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


def next_k_sample_indices(start_idx: int, end_idx: int, n_samples: int, k=16) -> np.ndarray:
    """
    从指定索引出发，按步长为 k 进行采样，确保采样索引不超出指定范围，
    如果采样数量不足，从 end_idx - 1 填充。

    参数：
    - start_idx: 采样的起始索引
    - end_idx: 采样的结束索引（当前索引）
    - n_samples: 需要采样的样本数
    - k: 步长

    返回：
    - indices: 采样的索引数组，按升序排列
    """
    # 计算从 start_idx 到 end_idx-1 的所有索引，步长为 k
    indices = np.arange(start_idx, end_idx, k)
    
    # 如果采样数量不足，则填充 end_idx - 1，直到数量为 n_samples
    if len(indices) < n_samples:
        extra = np.full(n_samples - len(indices), end_idx - 1)  # 填充 end_idx - 1
        indices = np.concatenate((indices, extra))  # 合并补充的索引
    
    # 如果采样数量超过 n_samples，进行截断
    indices = indices[:n_samples]
    
    # 排序，确保返回的索引按升序排列
    indices = np.sort(indices)

    return indices

def exponent_sample_indices(start_idx: int, end_idx: int, n_samples: int) -> np.ndarray:
    """
    从指定索引出发，按指数级偏移采样索引，确保采样索引不超出指定范围，
    超出范围时统一取 end_idx - 1。

    参数：
    - start_idx: 采样的起始索引
    - end_idx: 采样的结束索引（当前索引）
    - n_samples: 需要采样的样本数

    返回：
    - indices: 采样的索引数组，按升序排列
    """
    # 生成指数偏移序列，从 2^4 开始，逐步增加
    powers = 2 ** (4 + np.arange(n_samples))  # 从 2^4 开始，依次为 2^4, 2^5, ...
    
    # 计算采样的索引
    exp_seq = start_idx + powers  # 从 start_idx 开始，按指数级偏移
    
    # 如果超出了 end_idx - 1，则统一取 end_idx - 1
    exp_seq = np.minimum(exp_seq, end_idx - 1)

    # 排序，确保返回的索引按升序排列
    exp_seq = np.sort(exp_seq)

    return exp_seq


# @numba.jit(nopython=True)
# def left_right_sample_indices(start_idx: int, end_idx: int, n_samples: int, direction: str) -> np.ndarray:
    

#     def binary_sample_indices_long(start_idx: int, end_idx: int, n_samples: int, direction: str = 'right') -> np.ndarray:
#         """
#         偏左边界或偏右边界的二分采样索引
#         :param start_idx: 起始索引
#         :param end_idx: 结束索引（不包括）
#         :param n_samples: 采样数
#         :param direction: 'left' 为偏左边界二分，'right' 为偏右边界二分
#         :return: 采样的索引数组
#         """
#         indices = np.zeros(n_samples, dtype=np.int64)
#         assert n_samples > 0, "n_samples must be positive"

#         left = start_idx
#         right = end_idx - 1

#         cnt = 0
#         orimid = (left + right) // 2  # 初始 mid

#         while cnt < n_samples:
#             # 当前区间长度
#             section_len = right - left + 1
#             # 如果当前区间长度小于等于剩余采样数，则直接将剩余的采样点填充为边界
#             remain = n_samples - cnt
#             if section_len <= remain:
#                 indices[cnt:cnt + section_len] = np.arange(left, right + 1)
#                 cnt += section_len
#                 while cnt < n_samples:
#                     indices[cnt] = orimid  # 保证补充 mid 点
#                     cnt += 1
#                 break

#             # 计算中点
#             if direction == 'right':
#                 mid = left + (right - left + 1) // 2  # 偏右二分
#             elif direction == 'left':
#                 mid = left + (right - left) // 2  # 偏左二分
#             else:
#                 raise ValueError("direction must be either 'left' or 'right'")

#             indices[cnt] = mid
#             cnt += 1

#             if direction == 'right':
#                 left = mid + 1  # 偏右，移动左边界
#             elif direction == 'left':
#                 right = mid - 1  # 偏左，移动右边界

#         # 确保按升序排列
#         indices.sort()

#         return indices

#     def binary_sample_indices_short(start_idx: int, end_idx: int, n_samples: int, direction='left') -> np.ndarray:
#         """
#         从指定索引出发，按指数级偏移采样索引，确保采样索引不超出指定范围，
#         如果采样点不足，从最左或最右点进行填充（没有补0）。

#         参数：
#         - start_idx: 采样的起始索引
#         - end_idx: 采样的结束索引（当前索引）
#         - n_samples: 需要采样的样本数
#         - direction: 'left' 表示从当前索引往左偏移，'right' 表示从当前索引往右偏移

#         返回：
#         - indices: 采样的索引数组，按升序排列
#         """
#         # 生成指数偏移序列
#         powers = 2 ** np.arange(n_samples)  # 生成 2^i 的序列
#         powers = powers - 1  # 减去1确保从 0 开始

#         # 右采样：从 start_idx 开始，指数级偏移至 end_idx - 1
#         if direction == 'right':
#             exp_seq = start_idx + powers  # 从 start_idx 开始，按指数级偏移
#             exp_seq = exp_seq[exp_seq <= end_idx - 1]  # 确保不超出 end_idx - 1

#         # 左采样：从 end_idx 开始，指数级偏移至 start_idx
#         elif direction == 'left':
#             exp_seq = end_idx - 1 - powers  # 从 end_idx 开始，按指数级偏移
#             exp_seq = exp_seq[exp_seq >= start_idx]  # 确保不超出 start_idx

#         # 如果采样数量不够，进行填充，补充最右端或最左端点
#         if len(exp_seq) < n_samples:
#             remaining_samples = n_samples - len(exp_seq)
#             if direction == 'right':
#                 # 从 end_idx - 1 补充
#                 extra = np.full(remaining_samples, end_idx - 1)  # 用 end_idx - 1 填充
#             elif direction == 'left':
#                 # 从 start_idx 补充
#                 extra = np.full(remaining_samples, start_idx)  # 用 start_idx 填充
#             exp_seq = np.concatenate((exp_seq, extra))  # 合并补充的索引

#         # 排序，确保返回的索引按升序排列
#         exp_seq = np.sort(exp_seq)

#         return exp_seq


#     n_long = n_samples // 2
#     n_short = n_samples - n_long
#     long_indices = binary_sample_indices_long(start_idx, end_idx, n_long, direction)
#     short_indices = binary_sample_indices_short(start_idx, end_idx, n_short, direction)

#     return np.concatenate((long_indices, short_indices))
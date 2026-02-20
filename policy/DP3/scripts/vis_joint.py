import sys
import numpy as np
from collections import Counter

sys.path.append("./3D-Diffusion-Policy")
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

# zarr_path = "/home/liaohaoran/code/RoboTwin/policy/DP3/data/beat_block_hammer_loop_real-loop1-8-all-10hz-100.zarr"
zarr_path = "/home/liaohaoran/code/RoboTwin/policy/DP3/data/shake_bottle_loop_real-loop1-8-all-10hz-150.zarr"

keys_to_load = ["state", "instruction_int"]
replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys_to_load)

episode_ends = replay_buffer.episode_ends[:]
episode_start = new_arr = np.concatenate([[0], episode_ends[:-1]])

print(episode_ends-episode_start)
print(replay_buffer["instruction_int"][episode_start][...,0])

count_result = Counter(replay_buffer["instruction_int"][episode_start][...,0])
# 按数字大小排序并打印结果
for num in sorted(count_result):
    print(f"数字 {num} 出现了 {count_result[num]} 次")

for i in range(len(episode_start)):
    start_index = episode_start[i]
    end_index = episode_ends[i]
    data = replay_buffer["state"][start_index: end_index]
    print(data.shape)
    break
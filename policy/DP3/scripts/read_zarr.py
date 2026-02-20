from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

zarr_path = "/home/liaohaoran/code/RoboTwin/policy/DP3/data/shake_bottle_loop-loop1-8-all-200.zarr"
keys_to_load = ["instruction_int"]
replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys_to_load)
# 读取给定路径的一个HDF5文件
# 提取其中的joint_action/vector
# 并将其保存为NumPy数组文件(.npy)

import h5py
import numpy as np
import os

def hdf52npys(h5_file, npy_save_path_dir):
    """
    从HDF5文件中提取joint_action/vector数据并保存为.npy文件
    
    Args:
        h5_file: HDF5文件的路径
        npy_save_path_dir: 保存.npy文件的目录
    """
    # 检查文件是否存在
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
    
    # 打开HDF5文件
    with h5py.File(h5_file, 'r') as f:
        # 检查数据集是否存在
        if 'joint_action' not in f or 'vector' not in f['joint_action']:
            raise KeyError(f"'joint_action/vector' not found in {h5_file}")
        
        # 提取数据
        joint_action_vector = f['joint_action']['vector'][:]
        
        # 生成输出文件名（将.hdf5或.h5替换为.npy）
        output_file = os.path.splitext(h5_file)[0] + '_joint_action.npy'
        output_file = os.path.join(npy_save_path_dir, os.path.basename(output_file))

        # 保存为NumPy数组
        np.save(output_file, joint_action_vector)
        
        print(f"Successfully converted: {h5_file}")
        print(f"Data shape: {joint_action_vector.shape}")
        print(f"Saved to: {output_file}")

if __name__ == "__main__":
    task_name = "shake_bottle_loop"
    task_cfg = "loop3"
    episode_id = 0
    # 示例路径（请根据实际情况修改）
    h5_file_path = f"/home/liaohaoran/code/RoboTwin/data/{task_name}/{task_cfg}/data/episode{episode_id}.hdf5"
    npy_save_path_dir = f"/home/liaohaoran/code/RoboTwin/data/utils/data_for_analyse/{task_name}/{task_cfg}/"
    os.makedirs(npy_save_path_dir, exist_ok=True)
    output_path = hdf52npys(h5_file_path, npy_save_path_dir)
    print(f"\n✓ Conversion completed successfully!")

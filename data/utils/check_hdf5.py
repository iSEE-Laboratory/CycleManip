# type: ignore
# 读取HDF5文件并打印内容
import os
import pickle
import h5py
import numpy as np
from termcolor import cprint


def print_dataset_stats(name, data):
    """打印数据集的统计信息。
    
    Args:
        name: 数据集名称
        data: 数据数组
    """
    cprint(f"\n{'='*80}", "cyan")
    cprint(f"Dataset: {name}", "yellow", attrs=["bold"])
    cprint(f"{'='*80}", "cyan")
    
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Length: {len(data)}")
    
    # 对于数值类型数据，计算统计信息
    if np.issubdtype(data.dtype, np.number):
        print(f"  Mean: {np.mean(data):.6f}")
        print(f"  Std: {np.std(data):.6f}")
        print(f"  Min: {np.min(data):.6f}")
        print(f"  Max: {np.max(data):.6f}")
        print(f"  Median: {np.median(data):.6f}")
    elif data.dtype.kind == 'S':  # 字节串类型（编码的图像）
        cprint(f"  Type: Encoded images (JPEG)", "magenta")
        print(f"  Number of images: {len(data)}")
        if len(data) > 0:
            print(f"  Encoded size per image: ~{len(data[0])} bytes")
    else:
        cprint(f"  Type: {data.dtype}", "magenta")


def check_hdf5(hdf5_path):
    """读取并打印HDF5文件的详细信息。
    
    Args:
        hdf5_path: HDF5文件路径
    """
    if not os.path.exists(hdf5_path):
        cprint(f"Error: File not found: {hdf5_path}", "red")
        return
    
    cprint(f"\n{'#'*80}", "green", attrs=["bold"])
    cprint(f"Reading HDF5 file: {hdf5_path}", "green", attrs=["bold"])
    cprint(f"{'#'*80}\n", "green", attrs=["bold"])
    
    with h5py.File(hdf5_path, "r") as f:
        # 打印所有数据集名称
        cprint("Available datasets:", "cyan", attrs=["bold"])
        for key in f.keys():
            print(f"  - {key}")
        
        # 打印每个数据集的详细信息
        for key in f.keys():
            data = f[key][:]
            print_dataset_stats(key, data)
        
        # 特殊处理：如果有state和action，检查维度是否匹配
        if "state" in f.keys() and "action" in f.keys():
            state_shape = f["state"].shape
            action_shape = f["action"].shape
            cprint(f"\n{'='*80}", "green")
            cprint("Data consistency check:", "green", attrs=["bold"])
            cprint(f"{'='*80}", "green")
            print(f"  State shape: {state_shape}")
            print(f"  Action shape: {action_shape}")
            if state_shape[0] == action_shape[0]:
                cprint(f"  ✓ State and action lengths match: {state_shape[0]}", "green")
            else:
                cprint(f"  ✗ Length mismatch!", "red")
        
        # 打印文件总大小
        file_size = os.path.getsize(hdf5_path)
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"File size: {file_size / (1024**2):.2f} MB", "cyan", attrs=["bold"])
        cprint(f"{'='*80}\n", "cyan")


if __name__ == "__main__":
    # 示例：检查生成的HDF5文件
    hdf5_path = r"/home/liaohaoran/code/RoboTwin/data/beat_block_hammer/beat_block_hammer_clean/data/episode0.hdf5"
    check_hdf5(hdf5_path)


# 读取.hdf5文件
import h5py
import os

path = "/home/liaohaoran/code/RoboTwin/data/shake_bottle_loop_real/loop1-8/data/episode0.hdf5"

# 读取里面的 /joint_action/right_gripper 数据 as rg

# 改：当rg > 0.5
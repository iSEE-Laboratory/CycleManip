import h5py
import os
import numpy as np

def read_intrinsic(file_path):
    with h5py.File(file_path, 'r') as f:
        intrinsic_data = f["observation/head_camera/intrinsic_cv"][0]
    return intrinsic_data

if __name__ == "__main__":
    file_path = "/home/liaohaoran/code/RoboTwin/data/scan_object/general/data/episode8.hdf5"
    intrinsic = read_intrinsic(file_path)
    print(intrinsic)
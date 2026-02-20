import h5py

def print_h5_recursive_simple(file_path):
    """
    简洁的递归打印函数
    """
    def print_recursive(obj, prefix=""):
        for key in obj.keys():
            item = obj[key]
            full_path = f"{prefix}/{key}" if prefix else key
            
            if hasattr(item, 'shape'):
                # 数据集
                print(f"{full_path}: shape={item.shape}")
            else:
                # 组，递归处理
                print(f"{full_path}/")
                print_recursive(item, full_path)
    
    with h5py.File(file_path, 'r') as f:
        print("HDF5 file structure:")
        print_recursive(f)

if __name__ == "__main__":
    h5_file = "/home/liaohaoran/code/RoboTwin/data/morse_sos/general/data/episode0.hdf5"
    print_h5_recursive_simple(h5_file)
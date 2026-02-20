import numpy as np

def read_npz(file_path):
    try:
        data = np.load(file_path)
        print(f"Keys in the npz file: {list(data.keys())}")
        for key in data.keys():
            array = data[key]
            print(f"Key: {key}")
            print(f"Shape: {array.shape}")
            print(f"Mean: {np.mean(array):.4f}")
            print(f"Variance: {np.var(array):.4f}")
            print("-" * 40)
    except Exception as e:
        print(f"Error reading npz file: {e}")

if __name__ == "__main__":
    file_path = "/data1/haoran/code/LoopBreaker/data/shake_bottle_loop/demo_loop_clean/data/episode0.npz"
    read_npz(file_path)
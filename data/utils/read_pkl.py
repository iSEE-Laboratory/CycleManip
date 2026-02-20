import pickle

def print_pkl_keys_and_shapes(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print("Keys and shapes in the PKL file:")
        for key in data.keys():
            item = data[key]
            if hasattr(item, 'shape'):
                print(f"{key}: shape={item.shape}")
            else:
                print(f"{key}: (no shape attribute)")

if __name__ == "__main__":
    pkl_file = "/home/liaohaoran/code/RoboTwin/data/shake_bottle_loop/demo_loop_clean/.cache/episode0/0.pkl"
    print_pkl_keys_and_shapes(pkl_file)
import numpy
import os
import h5py

def hdf52npys(h5_file):
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"The file {h5_file} does not exist.")
    
    def recursive_extract(group, path=""):
        npz_data = {}
        for key in group.keys():
            item = group[key]
            current_path = f"{path}/{key}" if path else key
            if isinstance(item, h5py.Group):
                print(f"Entering group {current_path}")
                npz_data.update(recursive_extract(item, current_path))
            elif hasattr(item, 'shape'):
                npz_data[current_path] = numpy.array(item)
                print(f"Added {current_path} to npz data")
            else:
                print(f"{current_path}: (no shape attribute, skipping)")
        return npz_data

    with h5py.File(h5_file, 'r') as f:
        npz_data = recursive_extract(f)

    npz_dir = os.path.join(os.path.dirname(h5_file), "npzData")
    os.makedirs(npz_dir, exist_ok=True)
    npz_file = os.path.join(npz_dir, os.path.splitext(os.path.basename(h5_file))[0] + ".npz")
    numpy.savez(npz_file, **npz_data)
    print(f"Saved data to {npz_file}")

if __name__ == "__main__":
    # h5_file = "/data1/haoran/code/LoopBreaker/data/shake_bottle_loop/demo_loop_clean/data/episode0.hdf5"
    # hdf52npys(h5_file)
    dir = "/data1/haoran/code/LoopBreaker/data/shake_bottle_loop/demo_loop_clean/data/"
    for file in os.listdir(dir):
        if file.endswith(".hdf5"):
            h5_file = os.path.join(dir, file)
            hdf52npys(h5_file)
            print(f"Processed {file}")
            print("=" * 40)
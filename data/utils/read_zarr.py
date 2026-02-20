import zarr
import numpy as np
import os
from termcolor import cprint

def analyze_zarr_dataset(zarr_path):
    """
    Analyze a zarr dataset and print statistics for all variables.
    
    Args:
        zarr_path: Path to the zarr dataset
    """
    # Open the zarr store
    store = zarr.open(zarr_path, mode='r')
    
    print(f"Analyzing zarr dataset: {zarr_path}")
    print("=" * 80)

    # 打印出data/action的第一条数据
    if "data" in store and "state" in store["data"]:
        first_state = store["data"]["state"][0]
        print(f"First state data: {first_state}")
        print("-" * 80)

    if "data" in store and "action" in store["data"]:
        first_action = store["data"]["action"][0]
        print(f"First action data: {first_action}")
        print("-" * 80)

    if "data" in store and "instruction_int" in store["data"]:
        first_instruction_int = store["data"]["instruction_int"][0]
        print(f"First instruction_int data: {first_instruction_int}")
        print("-" * 80)

    if "data" in store and "instruction_sim" in store["data"]:
        first_instruction_sim = store["data"]["instruction_sim"][0]
        print(f"First instruction_sim data: {first_instruction_sim}")
        print("-" * 80)

    # 我们是用state预测state的，所以action是state的下一个时间步
    if "data" in store and "state" in store["data"] and "action" in store["data"]:
        # 这里assert一下， action[i] 应该等于 state[i+1]
        state_data = store["data"]["state"]
        action_data = store["data"]["action"]
        cnt = 0
        for i in range(len(action_data) - 1):
            try:
                assert np.allclose(action_data[i], state_data[i+1]), f"Action at index {i} does not match next state!"
            except AssertionError as e:
                cprint(e, "red")
                cnt += 1
        print(f"Number of mismatches: {cnt} out of {len(action_data)}")
        print("-" * 80)
    
    def analyze_group(group, prefix=""):
        """Recursively analyze zarr groups and arrays"""
        for key in group.keys():
            item = group[key]
            current_path = f"{prefix}/{key}" if prefix else key
            
            if hasattr(item, 'keys'):  # It's a group
                print(f"\nGroup: {current_path}")
                print("-" * 40)
                analyze_group(item, current_path)
            else:  # It's an array
                # Calculate statistics
                data = item[:]  # Load the data
                
                print(f"  Array: {current_path}")
                print(f"    Shape: {data.shape}")
                print(f"    Data type: {data.dtype}")
                
                if np.issubdtype(data.dtype, np.number):
                    # Only calculate numerical statistics for numeric data
                    print(f"    Min: {np.min(data):.6f}")
                    print(f"    Max: {np.max(data):.6f}")
                    print(f"    Mean: {np.mean(data):.6f}")
                    print(f"    Std: {np.std(data):.6f}")
                else:
                    print(f"    Non-numeric data - skipping numerical statistics")
                
                print()
    
    # Start analysis from root
    analyze_group(store)

if __name__ == "__main__":
    # Example usage
    zarr_path = "/home/liaohaoran/code/RoboTwin/policy/DP3/data/bbhl_demogen-loop1-5-200.zarr"
    analyze_zarr_dataset(zarr_path)
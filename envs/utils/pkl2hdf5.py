import h5py, pickle
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # Special handling for object_pos: each value might be a numpy array (the 7-tuple)
                if key == "object_pos":
                    # For object_pos, create a nested dict structure
                    parsed[key] = {}
                    for obj_name, obj_data in value.items():
                        if isinstance(obj_data, np.ndarray):
                            parsed[key][obj_name] = []
                        else:
                            parsed[key][obj_name] = []
                else:
                    parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Special handling for object_pos: create as a dataset group with each object's poses
            if key == "object_pos":
                object_pos_group = hdf5_group.create_group(key)
                for obj_name, pose_list in value.items():
                    pose_array = np.array(pose_list)
                    # If pose_array has object dtype (irregular entries), store elements individually
                    if pose_array.dtype == object:
                        sub = object_pos_group.create_group(obj_name)
                        for i, v in enumerate(pose_list):
                            try:
                                # numeric arrays can be stored directly
                                if isinstance(v, np.ndarray) and v.dtype != object:
                                    sub.create_dataset(str(i), data=v)
                                else:
                                    raise Exception('not numeric ndarray')
                            except Exception:
                                # fallback: pickle and store as vlen uint8
                                pickled = pickle.dumps(v)
                                vlen_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
                                ds = sub.create_dataset(str(i), (1,), dtype=vlen_uint8)
                                ds[:] = [np.frombuffer(pickled, dtype=np.uint8)]
                    else:
                        object_pos_group.create_dataset(obj_name, data=pose_array)
            else:
                subgroup = hdf5_group.create_group(key)
                create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            # Try to convert list into a reasonable HDF5-storable dataset.
            # Common cases: numeric lists, lists of numpy arrays with equal shape (stackable),
            # lists of bytes/strings (e.g. encoded images), or mixed/irregular lists.
            if len(value) == 0:
                # empty dataset
                hdf5_group.create_dataset(key, data=np.array([]))
                continue

            # Helper checks
            def all_numbers(lst):
                return all(np.isscalar(x) and not isinstance(x, (bytes, str)) for x in lst)

            def all_bytes_or_str(lst):
                return all(isinstance(x, (bytes, str)) for x in lst)

            def all_ndarray(lst):
                return all(isinstance(x, np.ndarray) for x in lst)

            if all_numbers(value):
                arr = np.array(value)
                hdf5_group.create_dataset(key, data=arr)
            elif all_bytes_or_str(value):
                # store binary blobs (e.g. encoded images) as vlen uint8 arrays
                vlen_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
                encoded = [x if isinstance(x, bytes) else str(x).encode() for x in value]
                dset = hdf5_group.create_dataset(key, (len(encoded),), dtype=vlen_uint8)
                dset[:] = [np.frombuffer(b, dtype=np.uint8) for b in encoded]
            elif all_ndarray(value):
                # check if stackable (same shape and numeric dtype)
                shapes = [x.shape for x in value]
                dtypes = [getattr(x, 'dtype', None) for x in value]
                if all(s == shapes[0] for s in shapes) and np.issubdtype(dtypes[0], np.number):
                    try:
                        arr = np.stack(value, axis=0)
                        hdf5_group.create_dataset(key, data=arr)
                    except Exception:
                        # fallback to storing each element as a subgroup
                        sub = hdf5_group.create_group(key)
                        for i, v in enumerate(value):
                            sub.create_dataset(str(i), data=v)
                else:
                    # irregular ndarray list: store each element separately under a subgroup
                    sub = hdf5_group.create_group(key)
                    for i, v in enumerate(value):
                        try:
                            sub.create_dataset(str(i), data=v)
                        except Exception:
                            # as ultimate fallback pickle the element into bytes and store as vlen uint8
                            pickled = pickle.dumps(v)
                            vlen_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
                            ds = sub.create_dataset(str(i), (1,), dtype=vlen_uint8)
                            ds[:] = [np.frombuffer(pickled, dtype=np.uint8)]
            else:
                # Mixed or unknown types: pickle each element and store as vlen uint8 arrays
                vlen_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
                pickled = [pickle.dumps(x) for x in value]
                dset = hdf5_group.create_dataset(key, (len(pickled),), dtype=vlen_uint8)
                dset[:] = [np.frombuffer(b, dtype=np.uint8) for b in pickled]
        else:
            # Handle scalar values and numpy arrays that slipped through
            # If it's a numpy array, ensure dtype is compatible
            if isinstance(value, np.ndarray):
                if value.dtype == object:
                    # store each element under a subgroup
                    sub = hdf5_group.create_group(key)
                    for i, v in enumerate(value.tolist()):
                        try:
                            if isinstance(v, np.ndarray) and v.dtype != object:
                                sub.create_dataset(str(i), data=v)
                            else:
                                raise Exception('not numeric ndarray')
                        except Exception:
                            pickled = pickle.dumps(v)
                            vlen_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
                            ds = sub.create_dataset(str(i), (1,), dtype=vlen_uint8)
                            ds[:] = [np.frombuffer(pickled, dtype=np.uint8)]
                else:
                    hdf5_group.create_dataset(key, data=value)
            else:
                # scalar or Python object
                if isinstance(value, (bytes, str)):
                    # store scalar bytes/str as single-element vlen uint8
                    data = value if isinstance(value, bytes) else value.encode()
                    vlen_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
                    ds = hdf5_group.create_dataset(key, (1,), dtype=vlen_uint8)
                    ds[:] = [np.frombuffer(data, dtype=np.uint8)]
                else:
                    try:
                        hdf5_group.create_dataset(key, data=value)
                    except Exception:
                        # fallback: store pickled value
                        pickled = pickle.dumps(value)
                        vlen_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
                        ds = hdf5_group.create_dataset(key, (1,), dtype=vlen_uint8)
                        ds[:] = [np.frombuffer(pickled, dtype=np.uint8)]


def pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    images_to_video(np.array(data_list["observation"]["head_camera"]["rgb"]), out_path=video_path)

    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_hdf5_video(folder_path, hdf5_path, video_path):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path)

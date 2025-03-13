# initial testing stuff/poking at the file

import sys
import joblib
import pickle
import numpy as np

def load_with_joblib_or_pickle(file_path):
    try:
        data = joblib.load(file_path)
        print(f"Successfully loaded '{file_path}' using joblib.load()!")
        return data, "joblib"
    except Exception as e:
        print(f"joblib.load() failed: {e}")
        print("Falling back to pickle.load()...")

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded '{file_path}' using pickle.load()!")
        return data, "pickle"
    except Exception as e:
        print(f"pickle.load() also failed: {e}")
        return None, None

def get_nested_list_shape(lst):
    shape = []
    while isinstance(lst, list) and len(lst) > 0:
        shape.append(len(lst))
        lst = lst[0]
    return tuple(shape)

def recursively_find_special_keys(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "video_frame":
                print("\nFound 'video_frame' key. Extracting last slice...")

                if isinstance(value, list):
                    # Convert the list to a NumPy array
                    try:
                        arr = np.array(value)
                    except Exception as e:
                        print(f"Failed to convert 'video_frame' list to np.array: {e}")
                        continue

                    if arr.ndim == 4 and arr.shape[0] > 0:
                        # Print shape and number of slices
                        print(f"video_frame: shape = {arr.shape}")
                        print(f"Number of slices in 'video_frame': {arr.shape[0]}")
                        last_slice = arr[-1, ...]  # slice out last
                        print(f"Last slice shape = {last_slice.shape}")
                        print("Full data (last slice):")
                        print(last_slice)
                    else:
                        print(f"'video_frame' is a list but shape after conversion is {arr.shape}; cannot slice.")
                else:
                    print(f"'video_frame' exists but is neither a list nor an np.ndarray; cannot slice.")

            elif key == "depth_frame":
                print("\nFound 'depth_frame' key. Extracting last slice...")

                if isinstance(value, list):
                    # Convert the list to a NumPy array
                    try:
                        arr = np.array(value)
                    except Exception as e:
                        print(f"Failed to convert 'depth_frame' list to np.array: {e}")
                        continue

                    if arr.ndim == 4 and arr.shape[0] > 0:
                        # Print shape and number of slices
                        print(f"depth_frame: shape = {arr.shape}")
                        print(f"Number of slices in 'depth_frame': {arr.shape[0]}")
                        last_slice = arr[-1, ...]  
                        print(f"Last slice shape = {last_slice.shape}")
                        print("Full data (last slice):")
                        print(last_slice)
                    else:
                        print(f"'depth_frame' is a list but shape after conversion is {arr.shape}; cannot slice.")
                else:
                    print(f"'depth_frame' exists but is neither a list nor an np.ndarray; cannot slice.")
            
            elif key == "pano_frame":
                print("\nFound 'pano_frame' key. Extracting last slice...")
                if isinstance(value, np.ndarray) and value.ndim == 4 and value.shape[0] > 0:
                    # Print shape and number of slices
                    print(f"pano_frame: shape = {value.shape}")
                    print(f"Number of slices in 'pano_frame': {value.shape[0]}")
                    last_slice = value[-1, ...]
                    print(f"Last slice shape = {last_slice.shape}")
                    print("Full data (last slice):")
                    print(last_slice)
                else:
                    print(f"'pano_frame' exists but shape is {getattr(value, 'shape', None)}; cannot slice.")
            
            elif key == "data_array":
                print("\nFound 'data_array' key. Extracting last row...")
                if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] > 0:
                    # Print shape and number of "slices" (rows in this case)
                    print(f"data_array: shape = {value.shape}")
                    print(f"Number of rows in 'data_array': {value.shape[0]}")
                    last_row = value[-1:, :]
                    print(f"Last row shape = {last_row.shape}")
                    print("Full data (last row):")
                    print(last_row)
                else:
                    print(f"'data_array' exists but shape is {getattr(value, 'shape', None)}; cannot slice.")
            
            else:
                recursively_find_special_keys(value)

    elif isinstance(data, (list, tuple)):
        for item in data:
            recursively_find_special_keys(item)

    else:
        pass

def main():
    file_path = "/Users/billygao/Downloads/V20HZVZS_V2DataNew_240105clark_shortC_lag"
    
    data, method_used = load_with_joblib_or_pickle(file_path)
    if data is None:
        print(f"Failed to read '{file_path}' using both joblib and pickle.")
        sys.exit(1)

    print(f"\nTop-level object type: {type(data).__name__}")
    print(f"Loaded with: {method_used}")

    if isinstance(data, dict):
        print("\nTop-level keys:")
        for key in data.keys():
            print(f"  {key}")

    recursively_find_special_keys(data)

if __name__ == "__main__":
    main()

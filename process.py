# pano frame processing

import sys
import os
import joblib
import numpy as np

def load_with_joblib_only(file_path):
    """
    Attempt to load file_path using joblib.load() only.
    Returns (data, "joblib") if successful, otherwise (None, None).
    """
    try:
        data = joblib.load(file_path)
        print(f"Successfully loaded '{file_path}' using joblib.load()!")
        return data, "joblib"
    except Exception as e:
        print(f"joblib.load() failed for '{file_path}': {e}")
        return None, None

def reorganize_data(data):
    """
    Given a dictionary with 'data_array' (n x 25) and 'pano_frame' (m x 180 x 360 x 11),
    reorganize each row of data_array into a structured dictionary:
      1)  time              -> index 0
      2)  position          -> indices 1..3
      3)  quaternion        -> indices 4..7
      4)  variance          -> indices 8..10
      5)  velocity          -> indices 11..13
      6)  angular_velocity  -> indices 14..16
      7)  joint_angles      -> indices 18..21
      8-18)  11 channels from pano_frame[i], 
             where i = data_array[n, 23].
             (red, green, blue, depth, class1..class7)

    Returns: a list of dictionaries, one per row in data_array.
    """

    # Extract data_array (n x 25) and pano_frame (m x 180 x 360 x 11)
    data_array = data["data_array"]  # shape: (n, 25)
    pano_frame = data["pano_frame"]  # shape: (m, 180, 360, 11)

    reorganized = []

    for row in data_array:
        time_val         = float(row[0])
        position         = row[1:4].astype(float)        # shape (3,)
        quaternion       = row[4:8].astype(float)        # shape (4,)
        variance         = row[8:11].astype(float)       # shape (3,)
        velocity         = row[11:14].astype(float)      # shape (3,)
        angular_velocity = row[14:17].astype(float)      # shape (3,)
        joint_angles     = row[18:22].astype(float)      # shape (4,)

        # index 23 => which pano_frame index to use
        i = int(row[23])
        slice_3d = pano_frame[i]  # shape (180, 360, 11)

        # Separate channels [0..10]
        red    = slice_3d[..., 0]
        green  = slice_3d[..., 1]
        blue   = slice_3d[..., 2]
        depth  = slice_3d[..., 3]
        class1 = slice_3d[..., 4]
        class2 = slice_3d[..., 5]
        class3 = slice_3d[..., 6]
        class4 = slice_3d[..., 7]
        class5 = slice_3d[..., 8]
        class6 = slice_3d[..., 9]
        class7 = slice_3d[..., 10]

        # Build a dictionary
        row_dict = {
            "time": time_val,
            "position": position,
            "quaternion": quaternion,
            "variance": variance,
            "velocity": velocity,
            "angular_velocity": angular_velocity,
            "joint_angles": joint_angles,
            "red": red,
            "green": green,
            "blue": blue,
            "depth": depth,
            "class1": class1,
            "class2": class2,
            "class3": class3,
            "class4": class4,
            "class5": class5,
            "class6": class6,
            "class7": class7,
        }

        reorganized.append(row_dict)

    return reorganized

def main():
    directory = "/Users/billygao/Downloads"
    files_processed = 0

    for filename in os.listdir(directory):
        if filename.startswith("eDS20HZVZS"):
            file_path = os.path.join(directory, filename)
            data, method_used = load_with_joblib_only(file_path)
            
            if data is None:
                print(f"Failed to read '{file_path}' using joblib; skipping.")
                print("-" * 60)
                continue

            print(f"\nTop-level object type: {type(data).__name__}")
            print(f"Loaded with: {method_used}")

            # Reorganize the data based on the instructions
            try:
                new_data = reorganize_data(data)
                instance_count = len(new_data)
                print(f"Number of instances in '{filename}': {instance_count}")

                # Save the reorganized data in the current directory
                out_name = filename + "_refined.joblib"
                out_path = os.path.join(os.getcwd(), out_name)
                joblib.dump(new_data, out_path, compress=True)
                print(f"Saved reorganized data to: {out_path}")
            except KeyError as ke:
                print(f"Error: Missing expected key in '{file_path}': {ke}")
            except Exception as e:
                print(f"Unexpected error processing '{file_path}': {e}")

            files_processed += 1
            print("-" * 60)

    print(f"\nTotal number of files processed: {files_processed}")

if __name__ == "__main__":
    main()

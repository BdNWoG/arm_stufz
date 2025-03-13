# video frame processing

import sys
import os
import joblib
import numpy as np

def load_with_joblib_only(file_path):
    try:
        data = joblib.load(file_path)
        print(f"Successfully loaded '{file_path}' using joblib.load()!")
        return data, "joblib"
    except Exception as e:
        print(f"joblib.load() failed for '{file_path}': {e}")
        return None, None

def reorganize_data(data):
    """
    Given a dictionary with:
      - 'data_array'  (n x 25)
      - 'video_frame' (m, 480, 848, 3)  # initially a list, converted to np.array

    For each row in data_array, reorganize into a structured dictionary:
      1)  time              -> index 0
      2)  position          -> indices 1..3
      3)  quaternion        -> indices 4..7
      4)  variance          -> indices 8..10
      5)  velocity          -> indices 11..13
      6)  angular_velocity  -> indices 14..16
      7)  joint_angles      -> indices 18..21
      8)  red, green, blue  -> from video_frame[i], where i = data_array[n, 23]

    Returns a list of dictionaries, one per row in data_array.
    """
    data_array = data["data_array"]

    video_list = data["video_frame"]
    video_frame = np.array(video_list)

    reorganized = []

    for row in data_array:
        time_val         = float(row[0])
        position         = row[1:4].astype(float)        # shape (3,)
        quaternion       = row[4:8].astype(float)        # shape (4,)
        variance         = row[8:11].astype(float)       # shape (3,)
        velocity         = row[11:14].astype(float)      # shape (3,)
        angular_velocity = row[14:17].astype(float)      # shape (3,)
        joint_angles     = row[18:22].astype(float)      # shape (4,)

        # index 23 => which frame index to use
        i = int(row[23])

        slice_rgb = video_frame[i]
        red       = slice_rgb[..., 0]
        green     = slice_rgb[..., 1]
        blue      = slice_rgb[..., 2]

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
        }

        reorganized.append(row_dict)

    return reorganized

def main():
    directory = "/Users/billygao/Downloads"
    files_processed = 0

    for filename in os.listdir(directory):
        if filename.startswith("V20HZVZS"):
            file_path = os.path.join(directory, filename)
            data, method_used = load_with_joblib_only(file_path)
            
            if data is None:
                print(f"Failed to read '{file_path}' using joblib; skipping.")
                print("-" * 60)
                continue

            print(f"\nTop-level object type: {type(data).__name__}")
            print(f"Loaded with: {method_used}")

            try:
                new_data = reorganize_data(data)
                instance_count = len(new_data)
                print(f"Number of instances in '{filename}': {instance_count}")

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

#!/usr/bin/env python3
import os
import joblib
import numpy as np

def compute_motion_features(
    data,
    T=2.0,                     # time window (seconds) for comparisons
    zero_speed_threshold=0.05, # "near zero" speed threshold
    speed_change_fraction=0.07, # fraction = 7% by default
    turn_angle_degrees=45.0    # heading change threshold
):
    """
      1) Stops       -> speed goes from non-zero to near-zero
      2) Starts      -> speed goes from near-zero to non-zero
      3) Non-steady  -> > 7% (configurable) change in speed
      4) Turning     -> > 45 degrees (configurable) change in heading
    """

    data_sorted = sorted(data, key=lambda x: x["time"])
    
    times = [row["time"] for row in data_sorted]
    positions = [row["position"] for row in data_sorted]
    
    n = len(times)
    velocities = [None] * n
    speeds = [0.0] * n
    headings = [None] * n   # unit direction vectors
    
    for i in range(1, n):
        dt = times[i] - times[i-1]
        if dt <= 0:
            continue
        displacement = positions[i] - positions[i-1]
        vel = displacement / dt
        velocities[i] = vel
        spd = np.linalg.norm(vel)
        speeds[i] = spd
        if spd > 1e-9:
            headings[i] = vel / spd  # direction unit vector
    
    def find_index_for_time(target_time, start_index):
        for idx in range(start_index, n):
            if times[idx] >= target_time:
                return idx
        return None
    
    features = {
        "stops": [],
        "starts": [],
        "non_steady": [],
        "turning": []
    }
    
    for i in range(1, n):
        t_i = times[i]
        j = find_index_for_time(t_i + T, i)
        if j is None:
            break
        
        speed_i = speeds[i]
        speed_j = speeds[j]
        
        if (speed_i > zero_speed_threshold) and (speed_j < zero_speed_threshold):
            features["stops"].append(t_i)
        
        if (speed_i < zero_speed_threshold) and (speed_j > zero_speed_threshold):
            features["starts"].append(t_i)
        
        if (speed_i > zero_speed_threshold) and (speed_j > zero_speed_threshold):
            ratio = speed_j / speed_i
            if (ratio < (1 - speed_change_fraction)) or (ratio > (1 + speed_change_fraction)):
                features["non_steady"].append(t_i)
        
        h_i = headings[i]
        h_j = headings[j]
        if (h_i is not None) and (h_j is not None):
            dot_val = np.dot(h_i, h_j)
            dot_val = max(-1.0, min(1.0, dot_val))
            angle_deg = np.degrees(np.arccos(dot_val))
            if angle_deg > turn_angle_degrees:
                features["turning"].append(t_i)
    
    return features


def main():
    directory = os.getcwd()
    motion_window = 1.0
    
    for filename in os.listdir(directory):
        if not filename.endswith("_refined.joblib"):
            continue
        
        file_path = os.path.join(directory, filename)
        
        try:
            data = joblib.load(file_path)
        except Exception as e:
            print(f"Could not load '{filename}': {e}")
            continue
        
        if not isinstance(data, list):
            print(f"Skipping '{filename}' because it's not a list of dictionaries.")
            continue
        
        features = compute_motion_features(data, T=motion_window)
        
        out_name = filename.replace(".joblib", "_motion_features.txt")
        out_path = os.path.join(directory, out_name)
        
        try:
            with open(out_path, "w") as f:
                f.write(f"File: {filename}\n")
                f.write(f"Motion events detected with a {motion_window}s window.\n")
                
                for key in ["stops", "starts", "non_steady", "turning"]:
                    times_list = features[key]
                    f.write(f"\n{key.upper()} ({len(times_list)} events):\n")
                    if times_list:
                        f.write(", ".join(map(str, times_list)) + "\n")
                    else:
                        f.write("(None)\n")
                
            print(f"Results saved to {out_path}")
        except Exception as e:
            print(f"Could not write to {out_path}: {e}")


if __name__ == "__main__":
    main()

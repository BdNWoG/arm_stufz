# this file outputs lists of times
import os
import joblib
import numpy as np

def extract_times_for_classes(data):
    class_times = {f"class{i}": [] for i in range(1, 8)}

    for row in data:
        for cidx in range(1, 8):
            class_name = f"class{cidx}"
            
            if class_name not in row:
                continue
            
            if np.any(row[class_name] != 0):
                class_times[class_name].append(row["time"])
    
    return class_times


def main():
    directory = os.getcwd()
    
    for filename in os.listdir(directory):
        if not filename.endswith("_refined.joblib"):
            continue
        
        file_path = os.path.join(directory, filename)
        
        try:
            data = joblib.load(file_path)
        except Exception as e:
            print(f"Could not load {filename} with joblib: {e}")
            continue
        if not isinstance(data, list):
            print(f"Skipping '{filename}' - expected a list of dictionaries after loading.")
            continue
        
        class_times = extract_times_for_classes(data)
        
        out_name = filename.replace(".joblib", "_classes_times.txt")
        out_path = os.path.join(directory, out_name)
        
        try:
            with open(out_path, "w") as f:
                f.write(f"File: {filename}\n")
                f.write("Time values for each class that has nonzero pixels:\n")
                for cidx in range(1, 8):
                    key = f"class{cidx}"
                    times_list = class_times[key]
                    f.write(f"\n{key} ({len(times_list)} frames):\n")
                    f.write(", ".join(map(str, times_list)) + "\n")
            print(f"Results saved to {out_path}")
        except Exception as e:
            print(f"Could not write to {out_path}: {e}")


if __name__ == "__main__":
    main()

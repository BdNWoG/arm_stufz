#!/usr/bin/env python3

import sys
import os
import joblib
import numpy as np

def main():
    if len(sys.argv) != 2:
        print("Provide exactly one argument (x from 1..7).")
        sys.exit(1)

    # Parse x and validate
    try:
        x = int(sys.argv[1])
        if x < 1 or x > 7:
            raise ValueError
    except ValueError:
        print(f"Invalid argument '{sys.argv[1]}'. Must be an integer from 1..7.")
        sys.exit(1)

    class_key = f"class{x}"
    print(f"Filtering for rows where '{class_key}' is NOT all zeros...")

    # We'll look for all *_refined.joblib files in the current directory
    current_dir = os.getcwd()
    refined_files = [f for f in os.listdir(current_dir) if f.endswith("_refined.joblib")]

    filtered_rows = []
    files_processed = 0
    rows_processed = 0
    rows_kept = 0

    for file_name in refined_files:
        file_path = os.path.join(current_dir, file_name)
        print(f"\nLoading refined data from: {file_path}")
        
        try:
            data_rows = joblib.load(file_path) 
            files_processed += 1
        except Exception as e:
            print(f"Failed to load '{file_name}': {e}")
            continue

        for row in data_rows:
            rows_processed += 1
            if np.any(row[class_key] != 0):
                filtered_rows.append(row)
                rows_kept += 1

    print("\nFiltering complete.")
    print(f"Total files processed: {files_processed}")
    print(f"Total rows processed: {rows_processed}")
    print(f"Total rows kept: {rows_kept}")

    # Save the filtered results
    out_filename = f"filtered_class{x}.joblib"
    out_path = os.path.join(current_dir, out_filename)
    try:
        joblib.dump(filtered_rows, out_path, compress=True)
        print(f"Filtered rows saved to: {out_path}")
    except Exception as e:
        print(f"Error saving filtered data to '{out_path}': {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

JOBLIB_FILE = "/Users/billygao/Downloads/eDS20HZVZS_V2DataRedo_realsense0801_lag_refined.joblib"

RIGHT_THIGH_IDX = 0
LEFT_THIGH_IDX  = 2

PEAK_HEIGHT_THRESHOLD = 0.1
PEAK_DISTANCE = 5

SLIDING_WINDOW = 5.0

def find_steps_in_angle(times, angles, height_threshold=0.1, distance=5):
    """
    Finds the times at which 'angles' has local peaks.
    Returns: step_times, step_indices
    """
    peak_indices, _ = find_peaks(
        angles,
        height=height_threshold,  # ignore small peaks
        distance=distance         # ignore peaks too close in the index domain
    )
    step_times = times[peak_indices]
    return step_times, peak_indices

def compute_step_metrics(rows):
    rows_sorted = sorted(rows, key=lambda r: r["time"])
    times = np.array([r["time"] for r in rows_sorted], dtype=float)

    right_thigh_angles = np.array([r["joint_angles"][RIGHT_THIGH_IDX] for r in rows_sorted], dtype=float)
    left_thigh_angles  = np.array([r["joint_angles"][LEFT_THIGH_IDX]  for r in rows_sorted], dtype=float)

    pos_x = np.array([r["position"][0] for r in rows_sorted], dtype=float)

    right_step_times, right_peak_idx = find_steps_in_angle(
        times, right_thigh_angles,
        height_threshold=PEAK_HEIGHT_THRESHOLD,
        distance=PEAK_DISTANCE
    )
    left_step_times, left_peak_idx = find_steps_in_angle(
        times, left_thigh_angles,
        height_threshold=PEAK_HEIGHT_THRESHOLD,
        distance=PEAK_DISTANCE
    )

    all_step_times = np.concatenate([right_step_times, left_step_times])
    all_step_indices = np.concatenate([right_peak_idx, left_peak_idx])

    sort_idx = np.argsort(all_step_times)
    step_times_merged = all_step_times[sort_idx]
    step_indices_merged = all_step_indices[sort_idx]

    if len(step_times_merged) < 2:
        return times, np.array([]), np.array([]), step_times_merged

    intervals = np.diff(step_times_merged) 
    freq_array = 1.0 / intervals 

    step_positions = pos_x[step_indices_merged]
    size_array = np.abs(np.diff(step_positions))

    return times, freq_array, size_array, step_times_merged

def main():
    if not os.path.exists(JOBLIB_FILE):
        print(f"Cannot find file: {JOBLIB_FILE}")
        return

    rows = joblib.load(JOBLIB_FILE)
    if not isinstance(rows, list) or len(rows) == 0:
        print("No valid rows found. Exiting.")
        return

    times_all, freq_array, size_array, step_times = compute_step_metrics(rows)

    rows_sorted = sorted(rows, key=lambda r: r["time"])
    total_frames = len(rows_sorted)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    plt.ion()

    current_frame = 0

    def update_left_subplot():
        row = rows_sorted[current_frame]
        r = np.asarray(row["red"])
        g = np.asarray(row["green"])
        b = np.asarray(row["blue"])
        base_img = np.dstack([r, g, b]).astype(float) / 255.0

        ax_left.clear()
        ax_left.imshow(base_img, origin='upper', interpolation='nearest')
        ax_left.set_title(f"Frame {current_frame+1}/{total_frames}\nTime={row['time']:.2f}s")

    def update_right_subplot():
        ax_right.clear()
        ax_right.set_title("Step Frequency & Size vs Time")
        ax_right.set_xlabel("Time (s)")

        if len(step_times) < 2:
            return

        t_current = rows_sorted[current_frame]["time"]
        t_start = t_current - SLIDING_WINDOW

        times_for_freq = step_times[1:]

        mask = (times_for_freq >= t_start) & (times_for_freq <= t_current)

        plot_times = times_for_freq[mask]
        plot_freqs = freq_array[mask]
        plot_sizes = size_array[mask]

        ax_right.plot(plot_times, plot_freqs, 'b-o', label="Step Freq (Hz)")
        ax_right.plot(plot_times, plot_sizes, 'r-o', label="Step Size (Δx)")

        ax_right.legend(loc='best')

        ax_right.set_xlim(t_start, t_current + 0.5) 
        ax_right.relim()
        ax_right.autoscale_view(True, True, True)

        ax_right.axvline(x=t_current, color='k', linestyle='--')

    def on_key(event):
        nonlocal current_frame
        if event.key == 'right':
            current_frame = (current_frame + 1) % total_frames
        elif event.key == 'left':
            current_frame = (current_frame - 1) % total_frames
        else:
            return 

        update_left_subplot()
        update_right_subplot()
        plt.draw()

    update_left_subplot()
    update_right_subplot()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)

if __name__ == "__main__":
    main()

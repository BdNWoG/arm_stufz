#!/usr/bin/env python3

import matplotlib.pyplot as plt
import joblib
import numpy as np

# EDIT THIS PATH:
JOBLIB_FILE = "/Users/billygao/Downloads/V20HZVZS_V2DataNew_240105clark_shortC_lag_refined.joblib"

def main():
    rows = joblib.load(JOBLIB_FILE)
    total_frames = len(rows)
    if total_frames == 0:
        print("No rows found in the loaded file. Exiting.")
        return

    fig, ax = plt.subplots()
    plt.ion()

    current_frame = 0
    current_class = 0 

    def update_plot():
        row = rows[current_frame]

        r = row["red"]
        g = row["green"]
        b = row["blue"]
        base_img = np.dstack([r, g, b]).astype(float) / 255.0

        ax.clear()
        ax.imshow(base_img, origin='upper', interpolation='nearest')

        if current_class != 0:
            class_key = f"class{current_class}"
            class_map = row[class_key]

            overlay = np.zeros((class_map.shape[0], class_map.shape[1], 4), dtype=float)
            overlay[..., 0] = 1.0   # Red channel
            overlay[..., 1] = 0.0   # Green channel
            overlay[..., 2] = 0.0   # Blue channel
            overlay[..., 3] = 0.8 * (class_map != 0)

            ax.imshow(overlay, origin='upper', interpolation='nearest')

        ax.set_title(f"Frame {current_frame+1}/{total_frames}, Class {current_class} (0=none)")
        plt.draw()

    def on_key(event):
        """Key press event handler for arrow keys."""
        nonlocal current_frame, current_class
        if event.key == 'left':
            current_class = (current_class - 1) % 8  # cycles 0..7
        elif event.key == 'right':
            current_class = (current_class + 1) % 8
        elif event.key == 'up':
            current_frame = (current_frame + 1) % total_frames
        elif event.key == 'down':
            current_frame = (current_frame - 1) % total_frames
        update_plot()

    fig.canvas.mpl_connect('key_press_event', on_key)

    update_plot()
    plt.show(block=True)

if __name__ == "__main__":
    main()

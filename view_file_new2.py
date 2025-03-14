#!/usr/bin/env python3

import matplotlib.pyplot as plt
import joblib
import numpy as np

# EDIT THIS PATH:
JOBLIB_FILE = "/Users/billygao/Downloads/V20HZVZS_V2DataNew_240105clark_shortC_lag_refined_first50_with_gdino_sam2.joblib"

CLASS_NAMES = [
    "original",   # 0
    "doors",      # 1
    "stairs",     # 2
    "curbs",      # 3
    "humans",     # 4
    "walls",      # 5
    "obstacles",  # 6
    "bikes",      # 7
    "cars",       # 8
]

def main():
    rows = joblib.load(JOBLIB_FILE)
    total_frames = len(rows)
    if total_frames == 0:
        print("No rows found in the loaded file. Exiting.")
        return

    fig, ax = plt.subplots()
    plt.ion()

    current_frame = 0
    current_class_index = 0

    def update_plot():
        """Clear the axes and redraw the current frame using the chosen class view."""
        row = rows[current_frame]

        base_img = np.dstack([row["red"], row["green"], row["blue"]]).astype(float) / 255.0

        ax.clear()

        class_name = CLASS_NAMES[current_class_index]
        if class_name == "original":
            ax.imshow(base_img)
            ax.set_title(f"Frame {current_frame+1}/{total_frames}, View = original")
        else:
            masked_key = f"{class_name}_masked"
            if masked_key not in row:
                print(f"Warning: '{masked_key}' not in row; showing base image instead.")
                ax.imshow(base_img)
                ax.set_title(f"Frame {current_frame+1}/{total_frames}, Missing '{masked_key}'")
            else:
                masked_img = row[masked_key].astype(float) / 255.0
                ax.imshow(masked_img)
                ax.set_title(f"Frame {current_frame+1}/{total_frames}, Class = {class_name}")

        plt.draw()

    def on_key(event):
        nonlocal current_frame, current_class_index
        if event.key == 'left':
            current_class_index = (current_class_index - 1) % len(CLASS_NAMES)
        elif event.key == 'right':
            current_class_index = (current_class_index + 1) % len(CLASS_NAMES)
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

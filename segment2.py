#!/usr/bin/env python3

import os
import sys
import joblib
import numpy as np
import torch
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert

# GroundingDINO & SAM2 Imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# ---------------------------------------------------------
# User Hyperparameters
# ---------------------------------------------------------
JOBLIB_FILE = "/Users/billygao/Downloads/V20HZVZS_V2DataNew_240105clark_shortC_lag_refined.joblib"

# For demonstration, we have multiple text prompts for bounding boxes:
TEXT_PROMPTS = [
    "door.",
    "stair.",
    "curb.",
    "human.",
    "wall.",
    "obstacle.",
    "bike.",
    "car.",
]

# If you want to pass them all in a single prompt to GroundingDINO, 
# you could do something like "doors. stairs. curbs. ..." 
# But it’s often better to do them individually or you can do advanced prompting.

SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT   = "./checkpoints/sam2.1_hiera_large.pt"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

# Box and text thresholds for GroundingDINO
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.15

DEVICE = "cpu"

print("Using device:", DEVICE)

OUTPUT_DIR = Path("outputs/joblib_sam2_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We'll only keep the first N frames for demonstration
MAX_FRAMES = 10


def single_mask_to_rle(mask: np.ndarray) -> dict:
    """Encode a binary mask to COCO RLE format (optional)."""
    rle = mask_util.encode(np.asfortranarray(mask[:, :, None].astype(np.uint8)))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def main():
    # 1) Load the .joblib data
    if not os.path.exists(JOBLIB_FILE):
        print(f"File not found: {JOBLIB_FILE}")
        sys.exit(1)

    rows = joblib.load(JOBLIB_FILE)
    if not isinstance(rows, list) or len(rows) == 0:
        print("Loaded data is empty or not a list. Exiting.")
        sys.exit(1)

    print(f"Original total frames: {len(rows)}")

    # 2) Keep only the first MAX_FRAMES
    rows = rows[:MAX_FRAMES]
    print(f"Truncating to the first {len(rows)} frames.")

    # 3) Build the SAM2 model/predictor
    print("Building SAM2 model...")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # 4) Build the GroundingDINO model
    print("Building GroundingDINO model...")
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )

    # 5) Process each of these frames
    for row_idx, row in enumerate(rows):
        # Reconstruct an RGB image from the joblib row
        red   = row["red"]
        green = row["green"]
        blue  = row["blue"]
        base_img = np.dstack([red, green, blue]).astype(np.uint8)
        tensor_img = torch.from_numpy(base_img).float()
        tensor_img = tensor_img.permute(2, 0, 1)

        # 6) For each text prompt, get bounding boxes from GroundingDINO
        for text_label in TEXT_PROMPTS:
            # a) Run predict()
            #    GroundingDINO likes the prompt to be lowercase + '.' 
            #    (some model variants do better with trailing period)
            #    e.g. "doors." or "stairs."
            full_prompt = f"{text_label}."

            boxes, confidences, labels = predict(
                model=grounding_model,
                image=tensor_img,         # real PIL image
                caption=full_prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE
            )

            # If no boxes found, skip
            if boxes is None or boxes.shape[0] == 0:
                # You can store an empty mask if you prefer
                print(f"No boxes found for '{text_label}' in frame {row_idx+1}.")
                row[f"{text_label.lower()}_masked"] = base_img.copy()  # or black image
                continue

            # 7) Convert boxes from cxcywh -> xyxy
            h, w, _ = base_img.shape
            boxes_abs = boxes * torch.tensor([w, h, w, h], device=boxes.device)
            input_boxes = box_convert(boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

            # 8) Pass image to SAM2
            sam2_predictor.set_image(base_img)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                # SAM2 call for each bounding box
                masks, sam_scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

            # masks shape => (N, H, W) or (N, 1, H, W) if there's more than one box
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            # 9) Combine or select the final mask
            #    If multiple boxes, you could combine them into a single mask
            #    or pick the "best" one. For now, let's combine with logical OR.
            combined_mask = np.zeros((h, w), dtype=bool)
            for m in masks:
                combined_mask |= m.astype(bool)

            # 10) Create a masked version
            masked_img = base_img.copy()
            masked_img[~combined_mask] = 0

            # 11) Store
            key_name = f"{text_label.lower()}_masked"
            row[key_name] = masked_img

        if (row_idx + 1) % 10 == 0:
            print(f"Processed {row_idx+1}/{len(rows)} frames...")

    # 12) Save the updated data
    out_name = os.path.splitext(JOBLIB_FILE)[0] + "_first50_with_gdino_sam2.joblib"
    joblib.dump(rows, out_name, compress=True)
    print(f"All done! Saved {len(rows)} frames (with GroundingDINO & SAM2 masks) to:\n{out_name}")

if __name__ == "__main__":
    main()
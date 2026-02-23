"""W&B sweep entry point for hyperparameter optimisation of the image pipeline."""

import os
import sys
import shutil
import tempfile

import cv2
import wandb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import process_image, classify_images

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "xray_images")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "classifier.model")


def run_pipeline():
    """Single sweep trial: process images, classify, log accuracy."""
    with wandb.init() as run:
        cfg = run.config

        # Create a temporary directory for this trial's processed images
        tmp_dir = tempfile.mkdtemp(prefix="sweep_results_")
        try:
            tags = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]

            for tag in tqdm(tags, desc=f"[{run.name}] Processing"):
                img_path = os.path.join(IMAGES_DIR, tag)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                if img is None:
                    tqdm.write(f"{tag} failed to load.")
                    continue

                processed = process_image(
                    img,
                    corner_quality=cfg.corner_quality,
                    border_width=cfg.border_width,
                    patch_size=cfg.patch_size,
                    dilation_size=cfg.dilation_size,
                    gamma=cfg.gamma,
                    clahe_clip_limit=cfg.clahe_clip_limit,
                    tile_grid_size=cfg.tile_grid_size,
                    apply_colour_contrast=cfg.apply_colour_contrast,
                )
                cv2.imwrite(os.path.join(tmp_dir, tag), processed)

            accuracy = classify_images(tmp_dir, MODEL_PATH)
            print(f"Accuracy: {accuracy:.4f}")
            wandb.log({"accuracy": accuracy})

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_pipeline()

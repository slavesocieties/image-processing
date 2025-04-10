import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# ------------- User config -------------
COCO_PATH     = "annotations.json"
IMAGE_DIR     = "sample_images/all"  # directory with original JPEGs
OUT_IMAGE_DIR = "preprocessed/images"
OUT_MASK_DIR  = "preprocessed/masks"
TARGET_SIZE   = (512, 512)  # (width, height)
# --------------------------------------

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# Load COCO-style annotations
with open(COCO_PATH, 'r') as f:
    coco = json.load(f)

# Map image_id to list of its annotations
img2anns = {}
for ann in coco['annotations']:
    img2anns.setdefault(ann['image_id'], []).append(ann)

# Filter to images that actually have annotations
annotated_images = [img for img in coco['images'] if img['id'] in img2anns]

print(f"Processing {len(annotated_images)} annotated images...")

for img_info in tqdm(annotated_images):
    fname   = img_info["file_name"]
    img_id  = img_info["id"]
    img_h   = img_info["height"]
    img_w   = img_info["width"]

    # Load image
    img_path = os.path.join(IMAGE_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Skipping missing image: {fname}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create empty mask
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Draw all polygons for this image
    for ann in img2anns[img_id]:
        # Each bbox is [min_x, min_y, width, height]
        x0, y0, w, h = ann["bbox"]
        x1 = x0 + w
        y1 = y0 + h
        
        # Create a rectangle polygon
        rect_pts = np.array([[x0, y0],
                            [x1, y0],
                            [x1, y1],
                            [x0, y1]], dtype=np.int32)

        # Fill the rectangle on the mask
        cv2.fillPoly(mask, [rect_pts], color=255)

    # Resize both image and mask
    img_resized  = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask,     TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    # Save to disk
    out_img_path  = os.path.join(OUT_IMAGE_DIR, fname)
    out_mask_path = os.path.join(OUT_MASK_DIR, os.path.splitext(fname)[0] + ".png")

    cv2.imwrite(out_img_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_mask_path, mask_resized)

print("✅ Done! Resized images and masks saved to 'preprocessed/'")

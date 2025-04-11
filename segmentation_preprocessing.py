import os
import json
import cv2
import boto3
import numpy as np
from tqdm import tqdm

# ------------- User config -------------
COCO_PATH     = "annotations.json"
IMAGE_DIR     = "sample_images"  # where images will be saved locally
OUT_IMAGE_DIR = "rect_preprocessed/images"
OUT_MASK_DIR  = "rect_preprocessed/masks"
TARGET_SIZE   = (640, 960)  # (width, height)

S3_BUCKET     = "ssda-production-jpgs"
S3_PREFIX     = ""  # prefix in S3 (e.g. 'ssda/images/')
# --------------------------------------

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# Set up S3 client
s3 = boto3.client('s3')

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

    img_path = os.path.join(IMAGE_DIR, fname)

    # --- Download image from S3 if not already present ---
    if not os.path.exists(img_path):
        s3_key = os.path.join(S3_PREFIX, fname).replace("\\", "/")
        try:
            s3.download_file(S3_BUCKET, s3_key, img_path)
        except Exception as e:
            print(f"⚠️  Failed to download {fname} from S3: {e}")
            continue

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Skipping unreadable image: {fname}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create empty mask
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Fill rectangles from bbox values
    for ann in img2anns[img_id]:
        if ann["category_id"] == 1:
            x0, y0, w, h = ann["bbox"]
            x1 = x0 + w
            y1 = y0 + h
            rect_pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
            cv2.fillPoly(mask, [rect_pts], color=255)  # use 255 for visibility

    # Resize both image and mask
    img_resized  = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask,     TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    # Save to disk
    out_img_path  = os.path.join(OUT_IMAGE_DIR, fname)
    out_mask_path = os.path.join(OUT_MASK_DIR, os.path.splitext(fname)[0] + ".png")

    cv2.imwrite(out_img_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_mask_path, mask_resized)

print(f"✅ Done! Downloaded, resized, and saved images + masks to {OUT_IMAGE_DIR[:OUT_IMAGE_DIR.rfind('/')]}")

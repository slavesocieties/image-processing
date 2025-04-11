import os
import json
import cv2
import boto3
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ------------- User config -------------
COCO_PATH     = "text_annotations.json"
IMAGE_DIR     = "sample_images"
OUT_IMAGE_DIR = "text_preprocessed/images"
OUT_MASK_DIR  = "text_preprocessed/masks"
TARGET_SIZE   = (640, 960)  # (width, height)

S3_BUCKET     = "ssda-production-jpgs"
S3_PREFIX     = ""
# --------------------------------------

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# Set up S3 client
s3 = boto3.client("s3")

# Load COCO-style annotations
with open(COCO_PATH, "r") as f:
    coco = json.load(f)

# Index images and annotations
id2img = {img["id"]: img for img in coco["images"]}
anns_by_img_cat = defaultdict(lambda: defaultdict(list))
for ann in coco["annotations"]:
    anns_by_img_cat[ann["image_id"]][ann["category_id"]].append(ann)

# Find maximum image ID with a text annotation
text_img_ids = [img_id for img_id, cats in anns_by_img_cat.items() if 2 in cats]
max_text_id = max(text_img_ids)

print(f"➡️  Will process all images with ID ≤ {max_text_id}")

for img_id in tqdm(sorted(id2img)):
    if img_id > max_text_id:
        continue

    img_info = id2img[img_id]
    fname = img_info["file_name"]
    img_path = os.path.join(IMAGE_DIR, fname)

    # Download image if needed
    if not os.path.exists(img_path):
        s3_key = os.path.join(S3_PREFIX, fname).replace("\\", "/")
        try:
            s3.download_file(S3_BUCKET, s3_key, img_path)
        except Exception as e:
            print(f"⚠️  Could not download {fname}: {e}")
            continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Could not read {fname}")
        continue

    # Find folio bbox
    folio_anns = anns_by_img_cat[img_id].get(1, [])           
    if not folio_anns:
        print(f"⚠️  No folio annotation for {fname}, skipping")
        continue

    fx, fy, fw, fh = map(int, folio_anns[0]["bbox"])    
    fx2, fy2 = fx + fw, fy + fh    
    cropped = img[fy:fy2, fx:fx2]

    # Build text mask
    mask = np.zeros((fh, fw), dtype=np.uint8)
    text_anns = anns_by_img_cat[img_id].get(2, [])
    for ann in text_anns:
        tx, ty, tw, th = map(int, ann["bbox"])
        rel_x = max(0, tx - fx)
        rel_y = max(0, ty - fy)
        rel_w = min(tw, fw - rel_x)
        rel_h = min(th, fh - rel_y)
        mask[rel_y:rel_y + rel_h, rel_x:rel_x + rel_w] = 255

    # Resize image + mask
    resized_img = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    # Save
    cv2.imwrite(os.path.join(OUT_IMAGE_DIR, fname), resized_img)
    cv2.imwrite(os.path.join(OUT_MASK_DIR, os.path.splitext(fname)[0] + ".png"), resized_mask)

print("✅ Done! Images and masks saved.")
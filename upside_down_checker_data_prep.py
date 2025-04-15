"""
Generate a training set for upright classification by sampling from S3,
performing folio segmentation, and applying optional 180° rotations.

Each run:
- Randomly selects an object key from the specified list
- Downloads it from the given S3 bucket
- Classifies folio count (1 or 2) using pretrained model
- Applies global orientation correction
- Segments the image into individual folios
- With 50% probability, rotates each folio 180°
- Saves each folio to the output directory
"""

import boto3
import random
import io
from pathlib import Path
from PIL import Image

from image_process_full import FolioCountModel, SegmentationModel

# CONFIGURATION ----------------------------------------------------------------
BUCKET = "ssda-production-jpgs"
OBJECT_KEYS = []  # will be populated once per session
MODEL_DIR = Path("./models")
OUTPUT_DIR = Path("./upright_training")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# INITIALIZE AWS CLIENT --------------------------------------------------------
s3 = boto3.client("s3")

# INITIALIZE MODELS ------------------------------------------------------------
folio_counter = FolioCountModel(MODEL_DIR / "folio_count_classifier.pth")
folio_seg     = SegmentationModel(MODEL_DIR / "unet_folio_segmentation.pth")

# UTILITY ----------------------------------------------------------------------
def list_all_s3_keys(bucket: str, prefix: str = "") -> list[str]:
    """Retrieve all object keys in an S3 bucket (efficient even for large buckets)."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    print(f"Discovered {len(keys):,} objects in '{bucket}'")
    return keys

def download_random_image() -> Image.Image:
    key = random.choice(OBJECT_KEYS)
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    img = Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
    print(f"Downloaded: {key}")
    return img, key

def fix_global_orientation(img: Image.Image, two_folios: bool) -> Image.Image:
    w, h = img.size
    if two_folios and h > w:
        return img.rotate(-90, expand=True)
    if not two_folios and w > h:
        return img.rotate(-90, expand=True)
    return img

# MAIN -------------------------------------------------------------------------
used_keys = set()

def main():
    global OBJECT_KEYS
    if not OBJECT_KEYS:
        index_file = Path("object_keys_remaining.txt")
        if index_file.exists():
            OBJECT_KEYS[:] = [line.strip() for line in index_file.read_text().splitlines() if line.strip()]
            print(f"Loaded {len(OBJECT_KEYS)} keys from saved file.")
        else:
            OBJECT_KEYS[:] = list_all_s3_keys(BUCKET)
            random.shuffle(OBJECT_KEYS)
            index_file.write_text("\n".join(OBJECT_KEYS))           

    key = OBJECT_KEYS.pop()    
    while key in used_keys:
        key = OBJECT_KEYS.pop()
    used_keys.add(key)
    page_img, key = download_random_image()
    stem = Path(key).stem
    label, _ = folio_counter.predict(page_img)
    two_folios = label == "two_folios"

    upright_img = fix_global_orientation(page_img, two_folios)

    # folio segmentation
    folios = []
    w, h = upright_img.size
    if two_folios:
        mid = w // 2
        overlap = int(0.05 * w)
        left_crop  = upright_img.crop((0, 0, mid + overlap, h))
        right_crop = upright_img.crop((mid - overlap, 0, w, h))
        lbox = folio_seg.predict_bbox(left_crop)
        rbox = folio_seg.predict_bbox(right_crop)
        if lbox:
            lx, ly, rx, ry = lbox
            folios.append(left_crop.crop((lx, ly, rx, ry)))
        if rbox:
            lx, ly, rx, ry = rbox
            folios.append(right_crop.crop((lx, ly, rx, ry)))
    else:
        bbox = folio_seg.predict_bbox(upright_img)
        if bbox:
            lx, ly, rx, ry = bbox
            folios.append(upright_img.crop((lx, ly, rx, ry)))

    if two_folios:
        suffixes = ["A", "B"]
        for i, folio in enumerate(folios):
            rotated = random.random() < 0.5
            if rotated:
                folio = folio.rotate(180, expand=False)
            tag = "rotated" if rotated else "upright"
            outname = OUTPUT_DIR / f"{stem}{suffixes[i]}-folio-{tag}.jpg"
            folio.save(outname)
            print(f"Saved: {outname.name}")            
    else:
        folio = folios[0]
        rotated = random.random() < 0.5
        if rotated:
            folio = folio.rotate(180, expand=False)
        tag = "rotated" if rotated else "upright"
        outname = OUTPUT_DIR / f"{stem}-folio-{tag}.jpg"
        folio.save(outname)
        print(f"Saved: {outname.name}")        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1, help="Number of samples to generate")
    args = parser.parse_args()
    for _ in range(args.count):
        main()
    Path("object_keys_remaining.txt").write_text("\n".join(OBJECT_KEYS))
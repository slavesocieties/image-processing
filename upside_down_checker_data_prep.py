import os
import random
from PIL import Image

def standardize_and_augment_orientation(directory_path, quality=85):
    """
    Applies orientation logic and re-saves every .jpg in the directory
    using consistent JPEG compression quality.
    """
    for filename in os.listdir(directory_path):
        if not filename.lower().endswith(".jpg"):
            continue

        file_path = os.path.join(directory_path, filename)
        try:
            img = Image.open(file_path).convert("RGB")

            # Rotate 90 degrees right if landscape
            if img.width > img.height:
                img = img.transpose(Image.ROTATE_270)

            # Random 50% chance to rotate 180 degrees
            if random.random() < 0.5:
                img = img.transpose(Image.ROTATE_180)

            # Always re-save with consistent compression settings
            img.save(file_path, quality=quality, optimize=True)
            print(f"Saved: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

standardize_and_augment_orientation("sample_images/one_folio")
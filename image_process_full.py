#!/usr/bin/env python3
"""
Alpha Image Processing Pipeline - full version
=============================================

This is the **complete, un-truncated** script that we have been incrementally
building.  It combines folio-count classification, global orientation fixes,
folio segmentation, per-folio upright checks, text-region segmentation,
optional deslant, and an optional visualisation output (`--viz`).  All
intermediate and final crops are saved at *original resolution* using the
filename conventions we agreed on.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

import matplotlib.pyplot as plt

DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE: Tuple[int, int] = (640, 960)  # (W,H) used during U‑Net training

def reflect_bbox_180(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
        """Reflect a bounding box 180° within a given image size (width x height).

        Parameters:
        - bbox: (x1, y1, x2, y2) bounding box
        - width: image width
        - height: image height

        Returns:
        - reflected bounding box (x1', y1', x2', y2')
        """
        x1, y1, x2, y2 = bbox
        new_x1 = width  - x2
        new_x2 = width  - x1
        new_y1 = height - y2
        new_y2 = height - y1
        return new_x1, new_y1, new_x2, new_y2

# ----------------------------------------------------------------------------
# Deslant helper --------------------------------------------------------------
# ----------------------------------------------------------------------------

def deslant_image(pil_img: Image.Image, angle_range: int = 10, step: float = 0.5) -> Image.Image:
    """
    Rotate the image to maximize alignment of text lines via projection profile contrast.
    """    

    img_rgb = np.array(pil_img)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_angle = 0
    best_score = -1

    for angle in np.arange(-angle_range, angle_range + step, step):
        M = cv2.getRotationMatrix2D((bw.shape[1] // 2, bw.shape[0] // 2), angle, 1.0)
        rotated = cv2.warpAffine(bw, M, (bw.shape[1], bw.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255)

        # Projection profile: row-wise black pixel count
        projection = np.sum(rotated == 0, axis=1)

        # Score = sum of squared differences (emphasizes sharp peaks)
        diffs = np.diff(projection)
        score = np.sum(diffs**2)

        if score > best_score:
            best_score = score
            best_angle = angle

    if abs(best_angle) < 0.5:
        return pil_img

    # Rotate original RGB image to best angle
    bg_mask = bw == 255
    avg_color = img_rgb[bg_mask].mean(axis=0) if np.any(bg_mask) else img_rgb.mean(axis=0)
    fill_color = tuple(int(c) for c in avg_color)

    return pil_img.rotate(best_angle, expand=True, fillcolor=fill_color)

# ----------------------------------------------------------------------------
# ResNet‑18 binary classifier wrapper ----------------------------------------
# ----------------------------------------------------------------------------
class ResNet18BinaryClassifier:
    def __init__(self, model_path: Path, class_names: List[str], device: str | None = None):
        self.device = torch.device(device or DEVICE_DEFAULT)
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = self._load(model_path)

    def _load(self, path: Path) -> torch.nn.Module:
        net = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Linear(net.fc.in_features, 2)
        net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        net.eval().to(self.device)
        return net

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> Tuple[str, float]:
        t = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(t)
        probs  = torch.softmax(logits, 1)[0]
        idx    = int(torch.argmax(probs))
        return self.class_names[idx], float(probs[idx])

class FolioCountModel(ResNet18BinaryClassifier):
    def __init__(self, path: Path, device: str | None = None):
        super().__init__(path, ["one_folio", "two_folios"], device)

class OrientationModel(ResNet18BinaryClassifier):
    def __init__(self, path: Path, device: str | None = None):
        super().__init__(path, ["right_side", "upside_down"], device)

# ----------------------------------------------------------------------------
# U‑Net segmentation wrapper --------------------------------------------------
# ----------------------------------------------------------------------------
class DoubleConv(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, 3, padding=1), torch.nn.BatchNorm2d(out_c), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_c, out_c, 3, padding=1), torch.nn.BatchNorm2d(out_c), torch.nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.seq(x)

class UNet(torch.nn.Module):
    """U-Net architecture *identical* to the one used in train_on_preprocessed_dataset.py."""
    def __init__(self, n_channels: int = 3, n_classes: int = 2):
        super().__init__()
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(512, 512))

        self.up1   = torch.nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 256)
        self.up2   = torch.nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 128)
        self.up3   = torch.nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 64)
        self.up4   = torch.nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc  = torch.nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5);  x = torch.cat([x, x4], dim=1); x = self.conv1(x)
        x  = self.up2(x);   x = torch.cat([x, x3], dim=1); x = self.conv2(x)
        x  = self.up3(x);   x = torch.cat([x, x2], dim=1); x = self.conv3(x)
        x  = self.up4(x);   x = torch.cat([x, x1], dim=1); x = self.conv4(x)
        return self.outc(x)

class SegmentationModel:
    def __init__(self, model_path: Path, device: str | None = None):
        self.device = torch.device(device or DEVICE_DEFAULT)
        self.model  = UNet().to(self.device)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def __init__(self, model_path: Path, device: str | None = None):
        self.device = torch.device(device or DEVICE_DEFAULT)
        self.model  = UNet().to(self.device)
                # Load state‑dict flexibly (handles DataParallel "module." prefixes)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] {model_path.name}: missing keys → {missing[:6]}…")
        if unexpected:
            print(f"[WARN] {model_path.name}: unexpected keys → {unexpected[:6]}…")
        self.model.eval()

    @torch.inference_mode()
    def predict_bbox(self, pil_img: Image.Image) -> Tuple[int,int,int,int] | None:
        w0,h0 = pil_img.size
        img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        resized = cv2.resize(img_np, IMG_SIZE, interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized.transpose(2,0,1)).float()[None]/255.0
        mask = torch.argmax(self.model(t.to(self.device)),1)[0].cpu().numpy().astype(np.uint8)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        x,y,w,h = cv2.boundingRect(max(cnts,key=cv2.contourArea))
        sx,sy = w0/IMG_SIZE[0], h0/IMG_SIZE[1]
        return int(x*sx), int(y*sy), int((x+w)*sx), int((y+h)*sy)

# ----------------------------------------------------------------------------
# Pipeline orchestrator -------------------------------------------------------
# ----------------------------------------------------------------------------
class AlphaPipeline:
    def __init__(self, model_dir: Path, outdir: Path, viz: bool = False):
        self.outdir = outdir
        self.viz = viz
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.folio_counter = FolioCountModel(model_dir/"folio_count_classifier.pth")
        self.folio_seg    = SegmentationModel(model_dir/"unet_folio_segmentation.pth")
        self.orientation  = OrientationModel(model_dir/"upside_down.pth")
        self.text_seg     = SegmentationModel(model_dir/"unet_text_segmentation.pth")

    # ----------------------------------------------------
    def process_page(self, img_path: Path) -> None:
        page_img_orig = Image.open(img_path).convert("RGB")
        viz_data: Dict[str,Any] = {"orig": page_img_orig}

        # 1. folio count
        count_label,_ = self.folio_counter.predict(page_img_orig)
        two_folios = count_label == "two_folios"

        # 2. global orientation
        page_img = self._fix_global_orientation(page_img_orig, two_folios)
        viz_data["upright"] = page_img
        page_w, page_h = page_img.size        

        # 3. folio segmentation
        folios = self._segment_folios(page_img, two_folios)
        viz_boxes: List[Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int] | None]] = []

        base_id = img_path.stem  # XXXXXX-YYYY
        for suffix, _, folio_bbox in folios:
            unique_id = base_id + suffix

            # --- 3a: crop folio from page (original resolution) -----------------
            folio_img = page_img.crop(folio_bbox)                        

            # --- 4. upright check ----------------------------------------------
            ori_label, _ = self.orientation.predict(folio_img)            
            if ori_label == "upside_down":
                folio_img = folio_img.rotate(180, expand=False)
                page_img = page_img.rotate(180, expand=False)                                
                folio_bbox = reflect_bbox_180(folio_bbox, page_w, page_h)                
            folio_img.save(self.outdir / f"{unique_id}-folio.jpg")

            # --- 5. text‑region segmentation -----------------------------------
            rel_box = self.text_seg.predict_bbox(folio_img)   # in folio coords
            abs_text_box = None
            if rel_box is not None:
                lx, ty, rx, by = rel_box

                # crop, deslant, save
                text_crop = folio_img.crop((lx, ty, rx, by))
                text_crop = deslant_image(text_crop)
                text_crop.save(self.outdir / f"{unique_id}-text.jpg")                

                # offset to page‑global coordinates
                fx0, fy0, _, _ = folio_bbox
                abs_text_box = (lx + fx0, ty + fy0, rx + fx0, by + fy0)                

            # store boxes for later viz (folio box already page‑global)
            viz_boxes.append((folio_bbox, abs_text_box))

            # stash crops for visualisation
            if self.viz:
                viz_data.setdefault("folio_imgs", []).append(folio_img)
                viz_data.setdefault("text_imgs", []).append(text_crop if abs_text_box else None)

        # visualise
        if self.viz:
            self._visualise_page(img_path.stem, page_img, viz_boxes, viz_data)

    # ----------------------------------------------------
    def _fix_global_orientation(self, img: Image.Image, two_folios: bool) -> Image.Image:
        w,h = img.size
        if two_folios and h > w:
            return img.rotate(-90, expand=True)
        if not two_folios and w > h:
            return img.rotate(-90, expand=True)
        return img

    def _segment_folios(self, page_img: Image.Image, two_folios: bool):
        w,h = page_img.size
        if not two_folios:
            bbox = self.folio_seg.predict_bbox(page_img) or (0,0,w,h)
            return [("", page_img, bbox)]
        # two folios: split with 5% overlap around center
        mid = w // 2
        overlap = int(0.05 * w)
        left_crop = page_img.crop((0,0, mid+overlap, h))
        right_crop = page_img.crop((mid-overlap, 0, w, h))
        left_bbox  = self.folio_seg.predict_bbox(left_crop)
        right_bbox = self.folio_seg.predict_bbox(right_crop)
        # translate bboxes back to page coords
        folios = []
        if left_bbox:
            l,t,r,b = left_bbox
            folios.append(("A", left_crop, (l, t, r, b)))
        if right_bbox:
            l,t,r,b = right_bbox
            folios.append(("B", right_crop, (mid-overlap+l, t, mid-overlap+r, b)))
        return folios    

    # ----------------------------------------------------
    def _visualise_page(self, stem: str, upright_img: Image.Image, boxes, data):
        two_folios = len(boxes) == 2
        if two_folios:
            fig, axes = plt.subplots(2,3, figsize=(12,8))
        else:
            fig, axes = plt.subplots(1,4, figsize=(12,4))
        axes = axes.flatten()
        # panel 0 original
        axes[0].imshow(data["orig"]); axes[0].set_title("Original"); axes[0].axis("off")
        # panel 1 upright
        axes[1].imshow(upright_img); axes[1].set_title("Upright"); axes[1].axis("off")
        # overlay bboxes on upright copy
        overlay = upright_img.copy()
        draw = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
        for folio_box, text_box in boxes:
            cv2.rectangle(draw, folio_box[:2], folio_box[2:], (0,0,255), 3)
            if text_box:
                cv2.rectangle(draw, text_box[:2], text_box[2:], (255,0,0), 3)
        axes[2].imshow(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
        axes[2].set_title("BBoxes"); axes[2].axis("off")
        # remaining panels
        idx = 3
        if two_folios:
            for img in data.get("text_imgs", []):
                if img is not None:
                    axes[idx].imshow(img); axes[idx].axis("off")
                idx += 1
        else:
            img = data.get("text_imgs", [None])[0]
            if img is not None:
                axes[3].imshow(img); axes[3].axis("off")
        plt.tight_layout()
        plt.savefig(self.outdir/f"{stem}-viz.png", dpi=150)
        plt.close(fig)

# ----------------------------------------------------------------------------
# Helper: gather images -------------------------------------------------------
# ----------------------------------------------------------------------------

def gather_images(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*.jpg" if recursive else "*.jpg"
    return list(path.glob(pattern))

# ----------------------------------------------------------------------------
# CLI ------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Alpha folio-processing pipeline")
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument("--outdir", type=Path, default=Path("./output"))
    parser.add_argument("--modeldir", type=Path, default=Path("./models"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--viz", action="store_true", help="Save diagnostic PNGs")
    args = parser.parse_args()

    pipeline = AlphaPipeline(args.modeldir, args.outdir, viz=args.viz)
    imgs = gather_images(args.input, args.recursive)
    if not imgs:
        sys.exit("No JPG images found at given path.")
    for p in imgs:
        pipeline.process_page(p)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
train_folio_segmentation.py  –  v2
----------------------------------
✓ filters out un‑annotated images
✓ resizes each (image, mask) pair to TARGET_H × TARGET_W
✓ keeps target size user‑tweakable via constants
"""

# ---------- user‑tunable constants ----------
IMG_ROOT   = "./sample_images/all"                 # directory with all JPEGs
COCO_FILE  = "./annotations.json"
TARGET_H   = 512                        # ← change height here
TARGET_W   = 512                        # ← change width  here
BATCH_SIZE = 4
EPOCHS     = 10
LR         = 1e-4
NUM_WORKERS = 0                         # keep 0 for notebooks / Windows

# ---------- imports ----------
import os, json, cv2, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- dataset ----------
class FolioDataset(Dataset):
    """Returns (image_tensor, mask_tensor) after resizing to TARGET_H×TARGET_W."""
    def __init__(self, root_dir: str, coco_json: str):
        self.root_dir = root_dir
        with open(coco_json, "r") as f:
            coco = json.load(f)

        # build mapping image_id → list[annotation]
        self.img2anns = {}
        for ann in coco.get("annotations", []):
            self.img2anns.setdefault(ann["image_id"], []).append(ann)

        # keep only images that actually have annotations
        annotated_ids = set(self.img2anns.keys())
        self.images   = [img for img in coco["images"] if img["id"] in annotated_ids]

    # -------- helper --------
    @staticmethod
    def _poly_to_mask(h, w, polygons):
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            pts = np.asarray(poly, np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        info     = self.images[idx]
        img_path = os.path.join(self.root_dir, info["file_name"])

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = info["height"], info["width"]

        # build binary mask
        anns  = self.img2anns.get(info["id"], [])
        polys = [seg for ann in anns for seg in ann.get("segmentation", [])]
        mask  = self._poly_to_mask(h, w, polys)

        # ---------- resize both image and mask ----------
        img  = cv2.resize(img,  (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

        # numpy → torch
        img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t


# ---------- tiny U‑Net (unchanged) ----------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.seq(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))
        self.up1   = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 256)
        self.up2   = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 128)
        self.up3   = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 64)
        self.up4   = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc  = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5); x = torch.cat([x, x4], 1); x = self.conv1(x)
        x  = self.up2(x);  x = torch.cat([x, x3], 1); x = self.conv2(x)
        x  = self.up3(x);  x = torch.cat([x, x2], 1); x = self.conv3(x)
        x  = self.up4(x);  x = torch.cat([x, x1], 1); x = self.conv4(x)
        return self.outc(x)

# ---------- training loop ----------
def main():
    ds = FolioDataset(IMG_ROOT, COCO_FILE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=NUM_WORKERS)

    net = UNet().to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(net.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        net.train(); running = 0.0
        for imgs, masks in dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            loss = crit(net(imgs), masks)
            loss.backward(); opt.step()
            running += loss.item()
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss = {running/len(dl):.4f}")

    torch.save(net.state_dict(), "unet_folio_segmentation.pth")
    print("✔ saved to unet_folio_segmentation.pth")

    # quick demo
    img_np, mask_np = infer(net, os.path.join(IMG_ROOT, ds.images[0]["file_name"]))
    plt.subplot(1,2,1); plt.imshow(img_np);           plt.title("Input")
    plt.subplot(1,2,2); plt.imshow(mask_np, cmap="gray"); plt.title("Predicted")
    plt.tight_layout(); plt.show()

# ---------- inference helper ----------
def infer(model, img_path):
    model.eval()
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    t   = torch.from_numpy(img.transpose(2,0,1)).float()[None]/255.0
    with torch.no_grad():
        pred = model(t.to(DEVICE))
    mask = torch.argmax(pred, 1)[0].cpu().numpy()
    return img, mask

# ---------- entry point ----------
if __name__ == "__main__":
    # Windows / notebooks are safe with num_workers=0; if you ever
    # raise NUM_WORKERS, keep this guard in place.
    main()
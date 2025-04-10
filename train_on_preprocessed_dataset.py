import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------- Constants ----------
IMAGE_DIR     = "preprocessed/images"
MASK_DIR      = "preprocessed/masks"
IMG_SIZE      = (512, 512)  # height, width
BATCH_SIZE    = 4
EPOCHS        = 10
LR            = 1e-4
NUM_WORKERS   = 0  # can raise if you add __main__ guard
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset ----------
class PreprocessedFolioDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path  = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(fname)[0] + ".png")

        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0  # (C, H, W)
        mask_t = torch.from_numpy(mask).long()  # (H, W), values 0 or 1

        return img_t, mask_t

# ---------- Model (same U‑Net) ----------
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

# ---------- Training loop ----------
def train():
    ds = PreprocessedFolioDataset(IMAGE_DIR, MASK_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = UNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train(); total_loss = 0.0
        for imgs, masks in dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {total_loss / len(dl):.4f}")

    torch.save(model.state_dict(), "unet_folio_preprocessed.pth")
    print("✅ Model saved as unet_folio_preprocessed.pth")

    # Demo
    img_np, mask_np = infer(model, os.path.join(IMAGE_DIR, ds.filenames[0]))
    plt.subplot(1, 2, 1); plt.imshow(img_np);           plt.title("Image")
    plt.subplot(1, 2, 2); plt.imshow(mask_np, cmap="gray"); plt.title("Predicted Mask")
    plt.tight_layout(); plt.show()

# ---------- Inference helper ----------
def infer(model, img_path):
    model.eval()
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    t   = torch.from_numpy(img.transpose(2,0,1)).float()[None]/255.0
    with torch.no_grad():
        pred = model(t.to(DEVICE))
    mask = torch.argmax(pred, 1)[0].cpu().numpy()
    return img, mask

# ---------- Entry point ----------
if __name__ == "__main__":
    train()
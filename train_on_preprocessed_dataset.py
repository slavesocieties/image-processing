#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

# ---------- User-configurable constants ----------
IMAGE_DIR    = "text_preprocessed/images"
MASK_DIR     = "text_preprocessed/masks"
IMG_SIZE     = (640, 960)  # (H, W) or (W, H) – be consistent with usage
BATCH_SIZE   = 2
EPOCHS       = 10
LR           = 1e-4
VAL_SPLIT    = 0.1  # 10% of data for validation
TEST_SPLIT   = 0.1  # 10% of data for test
NUM_WORKERS  = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Dataset ----------
class PreprocessedFolioDataset(Dataset):
    """
    Expects:
      preprocessed/images/<image_name>.jpg
      preprocessed/masks/<image_name>.png

    Returns:
      (img_t, mask_t), where:
        img_t is float32, shape (3, H, W), in [0,1]
        mask_t is int64, shape (H, W), in {0,1}
    """
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.filenames = sorted(os.listdir(image_dir))  # e.g. ["xxx.jpg", "yyy.jpg", ...]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path  = os.path.join(self.image_dir, fname)
        mask_name = os.path.splitext(fname)[0] + ".png"  # e.g. "xxx.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Remap 255 → 1 if needed
        # (If your mask is already 0 or 1, you can skip this step)
        mask = (mask > 0).astype(np.uint8)

        # Convert to torch tensors
        img_t  = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask).long()

        return img_t, mask_t


# ---------- U-Net model (unchanged) ----------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.seq(x)

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

        x  = self.up1(x5);  x = torch.cat([x, x4], dim=1); x = self.conv1(x)
        x  = self.up2(x);   x = torch.cat([x, x3], dim=1); x = self.conv2(x)
        x  = self.up3(x);   x = torch.cat([x, x2], dim=1); x = self.conv3(x)
        x  = self.up4(x);   x = torch.cat([x, x1], dim=1); x = self.conv4(x)
        return self.outc(x)


# ---------- Dice Evaluation ----------
@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Returns the average Dice score across 'dataloader'.
    We interpret class=1 as 'foreground' / 'folio'.
    """
    model.eval()
    total_dice = 0.0
    count = 0

    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)              # shape: (B,2,H,W)
        preds  = torch.argmax(logits, 1)  # shape: (B,H,W), values in {0,1}

        # Dice = 2*(preds & masks)/(preds+masks)
        intersection = (preds & masks).float().sum((1,2))
        union        = (preds | masks).float().sum((1,2))
        dice         = (2.0 * intersection + 1e-5) / (union + 1e-5)
        total_dice  += dice.sum().item()
        count       += len(dice)

    return total_dice / count


# ---------- Training ----------
def train_model():
    # 1) Load the full dataset
    full_ds = PreprocessedFolioDataset(IMAGE_DIR, MASK_DIR)
    total   = len(full_ds)
    val_sz  = int(VAL_SPLIT  * total)
    test_sz = int(TEST_SPLIT * total)
    train_sz = total - val_sz - test_sz

    # 2) Split
    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [train_sz, val_sz, test_sz],
        generator=torch.Generator().manual_seed(42)
    )

    # 3) DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train set size: {train_sz}, Val set size: {val_sz}, Test set size: {test_sz}")

    # 4) Model, loss, optimizer
    model = UNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 5) Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 6) Validation each epoch
        train_loss = total_loss / len(train_loader)
        val_dice   = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")

    # 7) Final test evaluation
    test_dice = evaluate(model, test_loader, DEVICE)
    print(f"Test Dice: {test_dice:.4f}")

    # 8) Save model
    torch.save(model.state_dict(), "unet_folio_split.pth")
    print("✅ Model saved as unet_folio_split.pth")

    # Optional: quick sample inference from test set
    test_sample_img, test_sample_mask = next(iter(test_loader))
    test_sample_img = test_sample_img.to(DEVICE)
    with torch.no_grad():
        sample_logits = model(test_sample_img)
    sample_preds = torch.argmax(sample_logits, dim=1).cpu().numpy()

    # Visualize the first sample from that batch
    img_np  = test_sample_img[0].cpu().numpy().transpose(1,2,0)
    pred_np = sample_preds[0]
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(img_np);            plt.title("Sample Input")
    plt.subplot(1,2,2); plt.imshow(pred_np, cmap="gray"); plt.title("Pred Mask")
    plt.tight_layout()
    plt.show()


# ---------- Entry Point ----------
if __name__ == "__main__":
    train_model()
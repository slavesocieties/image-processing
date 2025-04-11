import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------- Constants ----------
IMAGE_DIR     = "big_preprocessed/images"
MASK_DIR      = "big_preprocessed/masks"
IMG_SIZE      = (960, 640)  # height, width
BATCH_SIZE    = 2
EPOCHS        = 10
LR            = 1e-4
VAL_SPLIT     = 0.1
TEST_SPLIT    = 0.1
NUM_WORKERS   = 2
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Dataset ----------
class PreprocessedFolioDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path  = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(fname)[0] + ".png")

        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return img, mask


# ---------- Augmentations ----------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
    ToTensorV2()
])

eval_transform = A.Compose([
    A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
    ToTensorV2()
])


# ---------- Model ----------
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
        x = self.up1(x5); x = torch.cat([x, x4], 1); x = self.conv1(x)
        x = self.up2(x);  x = torch.cat([x, x3], 1); x = self.conv2(x)
        x = self.up3(x);  x = torch.cat([x, x2], 1); x = self.conv3(x)
        x = self.up4(x);  x = torch.cat([x, x1], 1); x = self.conv4(x)
        return self.outc(x)


# ---------- Dice Evaluation ----------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_dice = 0.0
    count = 0

    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, 1)

        intersection = (preds & masks).float().sum((1, 2))
        union        = (preds | masks).float().sum((1, 2))
        dice         = (2. * intersection + 1e-5) / (union + 1e-5)
        total_dice  += dice.sum().item()
        count       += len(dice)

    return total_dice / count


# ---------- Train ----------
def train():
    # Load full dataset
    base_ds = PreprocessedFolioDataset(IMAGE_DIR, MASK_DIR)

    # Split
    total     = len(base_ds)
    val_sz    = int(VAL_SPLIT * total)
    test_sz   = int(TEST_SPLIT * total)
    train_sz  = total - val_sz - test_sz
    train_ds, val_ds, test_ds = random_split(base_ds, [train_sz, val_sz, test_sz], generator=torch.Generator().manual_seed(42))

    # Assign transforms AFTER splitting
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform   = eval_transform
    test_ds.dataset.transform  = eval_transform

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model setup
    model = UNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_dice = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")

    test_dice = evaluate(model, test_loader, DEVICE)
    print(f"Test Dice Score: {test_dice:.4f}")

    torch.save(model.state_dict(), "unet_augmented_folio.pth")
    print("✅ Saved model as unet_augmented_folio.pth")

    # Quick visual check
    sample_img, _ = test_ds[0]
    model.eval()
    with torch.no_grad():
        pred = model(sample_img.unsqueeze(0).to(DEVICE))
    pred_mask = torch.argmax(pred, 1)[0].cpu().numpy()

    img_np = sample_img.permute(1, 2, 0).numpy()
    plt.subplot(1,2,1); plt.imshow(img_np); plt.title("Input")
    plt.subplot(1,2,2); plt.imshow(pred_mask, cmap="gray"); plt.title("Predicted Mask")
    plt.tight_layout(); plt.show()

# ---------- Entry ----------
if __name__ == "__main__":
    train()

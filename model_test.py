import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms

# ---------- CONFIG ----------
MODEL_PATH = "unet_text_segmentation.pth"
IMAGE_PATH = "text_preprocessed/images/106925-0156.jpg"  # <- change this
INPUT_SIZE = (640, 960)  # (width, height) — must match training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- U-Net Definition (simplified) ----------
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


# ---------- Load image and preprocess ----------
orig = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
resized = cv2.resize(orig, INPUT_SIZE, interpolation=cv2.INTER_AREA)
tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

# ---------- Load model ----------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with torch.no_grad():
    pred = model(tensor.to(DEVICE))
mask = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)

# ---------- Extract bounding box ----------
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vis = resized.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(vis, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)

# ---------- Show ----------
plt.figure(figsize=(10, 6))
plt.imshow(vis)
plt.title("Predicted Mask Bounding Box")
plt.axis("off")
plt.tight_layout()
plt.show()
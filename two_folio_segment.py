import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn

# ---------- CONFIG ----------
IMAGE_PATH = "sample_images/two_folio/DSC_0013.JPG"
MODEL_PATH = "unet_folio_split.pth"
IMG_SIZE = (960, 640)  # width x height (must match model)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPINE_OVERLAP = 40  # pixels to include around center when cropping left/right

# ---------- U-Net Definition ----------
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


# ---------- Helper: Segment one side ----------
def segment_side(model, img_side):
    resized = cv2.resize(img_side, IMG_SIZE, interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(resized.transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        pred = model(t)
    mask = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)
    # find bounding box
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return resized, mask, None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    crop = resized[y:y+h, x:x+w]
    return resized, mask, crop


# ---------- Load model ----------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------- Load and split image ----------
img = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
mid = W // 2
SPINE_OVERLAP = int(.05 * W)

left_crop  = img[:, :mid + SPINE_OVERLAP]
right_crop = img[:, mid - SPINE_OVERLAP:]

# ---------- Segment both halves ----------
left_img, left_mask, left_folio = segment_side(model, left_crop)
right_img, right_mask, right_folio = segment_side(model, right_crop)

# ---------- Display ----------
plt.figure(figsize=(12,6))
plt.subplot(2,2,1); plt.imshow(left_img);     plt.title("Left Half")
plt.subplot(2,2,2); plt.imshow(left_mask, cmap="gray"); plt.title("Left Predicted Mask")
plt.subplot(2,2,3); plt.imshow(right_img);    plt.title("Right Half")
plt.subplot(2,2,4); plt.imshow(right_mask, cmap="gray"); plt.title("Right Predicted Mask")
plt.tight_layout()
plt.show()

if left_folio is not None and right_folio is not None:
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(left_folio);  plt.title("Left Folio Crop")
    plt.subplot(1,2,2); plt.imshow(right_folio); plt.title("Right Folio Crop")
    plt.tight_layout(); plt.show()
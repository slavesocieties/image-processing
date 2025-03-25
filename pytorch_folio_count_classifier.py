import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import os

# --- SETTINGS ---
DATA_DIR = "sample_images"  # <-- change this to your dataset path
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

# --- DEVICE ---
device = torch.device("cpu")  # CPU-only version
torch.set_num_threads(8)

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # standard size for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

# --- DATASET AND LOADER ---
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes  # ['one_folio', 'two_folios']
print(f"Classes: {class_names}")

# Check class counts
from collections import Counter

counts = Counter(dataset.targets)
for class_idx, count in counts.items():
    print(f"{class_names[class_idx]}: {count}")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() - 1)

# --- MODEL ---
# Use the most up-to-date default ImageNet weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# --- LOSS FUNCTION (with class weights) ---
# Adjust weights to counter imbalance (e.g., inverse frequency)
weight = torch.tensor([1.0, 10.0])  # [one_folio, two_folios]
criterion = nn.CrossEntropyLoss(weight=weight)

# --- OPTIMIZER ---
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- TRAINING LOOP ---
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

# --- SAVE MODEL ---
torch.save(model.state_dict(), "folio_classifier.pth")
print("Model saved to folio_classifier.pth")
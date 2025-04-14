import torch
from torchvision import models, transforms
from PIL import Image
import os

from torchvision.models import resnet18, ResNet18_Weights

# Set up the same transform used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Map class indices back to labels
CLASS_NAMES = ['rightside_up', 'upside_down']

def predict_folio_class(image_path, model_path="upside_down.pth"):
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Load model
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)  # loads architecture
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        label = CLASS_NAMES[predicted.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

    return label, confidence

# standard single: 66450-0195.jpg
# single but rotated: 701241-0144.jpg
# standard double: 31510-0107.jpg
# rotated double: 770016-0392.jpg
img_path = "sample_images/102624-0240.jpg"
label, conf = predict_folio_class(img_path)
print(f"Predicted: {label} (Confidence: {conf:.2%})")

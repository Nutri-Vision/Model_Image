import json
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# ------------------ Settings ------------------ #
IMAGE_PATH = "./examples/appl.webp"   
set=1 
if(set==0):         
 MODEL_PATH = "./model1/best_model.pth"  # Your trained model
 CLASS_MAP_PATH = "./model1/class_mapping.json"  
else:
 MODEL_PATH = "./runs/exp_combined_small/best_model.pth"  # Your trained model
 CLASS_MAP_PATH = "./runs/exp_combined_small/class_mapping.json"  
    
IMAGE_SIZE = 224

# ------------------ Functions ------------------ #
def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size*1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def build_model(num_classes, dropout=0.5, pretrained=True):
    model = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model

def predict_image(model, img_path, class_idx_to_name, device, image_size=224):
    transform = get_transform(image_size)
    image = Image.open(img_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = probs.max(1)
        pred = pred.item()
        conf = conf.item()

    pred_class = class_idx_to_name[str(pred)]
    return pred_class, conf

# ------------------ Main ------------------ #
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Load class mapping
with open(CLASS_MAP_PATH, "r") as f:
    idx_to_class = json.load(f)

num_classes = len(idx_to_class)
model = build_model(num_classes, dropout=0.5, pretrained=False)

# Load checkpoint safely (backbone only if num_classes mismatch)
checkpoint = torch.load(MODEL_PATH, map_location=device)
checkpoint_dict = checkpoint["model_state"]

model_dict = model.state_dict()
# Update only matching keys (skip fc if shape mismatch)
pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model = model.to(device)

# Predict
pred_class, conf = predict_image(model, IMAGE_PATH, idx_to_class, device, image_size=IMAGE_SIZE)
print(f"Prediction: {pred_class} (confidence: {conf:.3f})")

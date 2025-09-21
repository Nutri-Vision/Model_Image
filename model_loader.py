import json
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

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

def predict_image(model, img_path, class_data, device, image_size=224):
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

    pred_class = class_data[str(pred)]["name"]
    nutrition = class_data[str(pred)]["nutrition"]
    return pred_class, conf, nutrition

def load_model_and_mapping(model_path, class_map_path, device):
    # Load class mapping with nutrition data
    with open(class_map_path, "r") as f:
        class_data = json.load(f)

    num_classes = len(class_data)
    model = build_model(num_classes, dropout=0.5, pretrained=False)

    # Load checkpoint safely
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_dict = checkpoint["model_state"]

    model_dict = model.state_dict()
    # Update only matching keys
    pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.to(device)
    return model, class_data
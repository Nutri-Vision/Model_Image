import json
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import requests

# ------------------ SETTINGS ------------------ #
IMAGE_PATH = "./examples/oml.webp"             # Image to test
MODEL_PATH = "./runs/exp_combined_small/best_model.pth"         # Your trained model
CLASS_MAP_PATH = "./runs/exp_combined_small/class_map1.json" # Mapping from indices to class names
IMAGE_SIZE = 224
API_KEY = " "                  # Replace with your USDA API key
CACHE_FILE = "nutrition_cache.json"            # Optional cache to avoid repeated API calls

# ------------------ FUNCTIONS ------------------ #
def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size*1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def build_model(num_classes, dropout=0.5, pretrained=False):
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

def get_nutrition_from_usda(food_name):
    """Query USDA API to get nutrition info for a food item."""
    params = {
        "api_key": API_KEY,
        "query": food_name,
        "pageSize": 1
    }
    response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("foods"):
            nutrients = data["foods"][0]["foodNutrients"]
            nutrition_info = {n["nutrientName"]: n["value"] for n in nutrients}
            return nutrition_info
    return None

# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load class mapping
    with open(CLASS_MAP_PATH, "r") as f:
        idx_to_class = json.load(f)

    # Load model
    num_classes = len(idx_to_class)
    model = build_model(num_classes, dropout=0.5, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    # Predict
    pred_class, conf = predict_image(model, IMAGE_PATH, idx_to_class, device, IMAGE_SIZE)
    print(f"\nPredicted Class: {pred_class} (Confidence: {conf:.3f})")

    # Load cache
    try:
        with open(CACHE_FILE, "r") as f:
            nutrition_cache = json.load(f)
    except:
        nutrition_cache = {}

    # Get nutrition info
    if pred_class in nutrition_cache:
        nutrition = nutrition_cache[pred_class]
    else:
        nutrition = get_nutrition_from_usda(pred_class)
        if nutrition:
            nutrition_cache[pred_class] = nutrition
            with open(CACHE_FILE, "w") as f:
                json.dump(nutrition_cache, f, indent=2)

    # Display nutrition
    if nutrition:
        print("\nNutrition Information:")
        for k, v in nutrition.items():
            print(f"{k}: {v}")
    else:
        print("\nNo nutrition info found for this food.")

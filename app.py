from flask import Flask, render_template, request
import torch
from PIL import Image
import os
import json
from train_food_classifier import build_model, predict_image

app = Flask(__name__)

# ------------------------
# CONFIGURATION
# ------------------------
MODEL_PATH = "./runs/exp1/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class mapping with nutrition
with open("./runs/exp1/class_mapping.json") as f:
    class_map = json.load(f)

# Build model and load weights
num_classes = len(class_map)
model = build_model(num_classes)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

# ------------------------
# ROUTES
# ------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    nutrition = None
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded!")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected!")

        # Save uploaded file temporarily
        os.makedirs("uploads", exist_ok=True)
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        # Predict class
        pred_class, conf = predict_image(model, filepath, class_map, DEVICE)
        fruit_info = class_map[str(pred_class)]  # ensure key is string
        fruit_name = fruit_info["name"]
        nutrition = fruit_info["nutrition"]
        prediction = f"{fruit_name} (confidence: {conf:.2f})"

    return render_template("index.html", prediction=prediction, nutrition=nutrition)

# ------------------------
# RUN APP
# ------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)

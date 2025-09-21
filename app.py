from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import json
import torch
from PIL import Image

import numpy as np
import torch


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and class mapping
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Update these paths according to your setup
MODEL_PATH = "./runs/exp_combined_small/best_model.pth"
CLASS_MAP_PATH = "./runs/exp_combined_small/class_mapping.json"

# Load class mapping
with open(CLASS_MAP_PATH, "r") as f:
    class_data = json.load(f)

# Import model functions
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

# Load the model
num_classes = len(class_data)
model = build_model(num_classes, dropout=0.5, pretrained=False)

# Load checkpoint safely
checkpoint = torch.load(MODEL_PATH, map_location=device)
checkpoint_dict = checkpoint["model_state"]

model_dict = model.state_dict()
# Update only matching keys
pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model = model.to(device)

# Meal history storage
meal_history = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict endpoint called")  # Debug
    if 'file' not in request.files:
        print("No file in request")  # Debug
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    print(f"File received: {file.filename}")  # Debug
    
    if file.filename == '':
        print("Empty filename")  # Debug
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {filepath}")  # Debug
        
        # Ensure upload directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        file.save(filepath)
        print("File saved successfully")  # Debug
        
        try:
            from PIL import Image
            pred_class, conf, nutrition = predict_image(model, filepath, class_data, device)
            
            # Calculate nutrition based on quantity
            quantity = float(request.form.get('quantity', 100))
            adjusted_nutrition = {
                'calories': round(nutrition['calories'] * (quantity / 100), 1),
                'protein': round(nutrition['protein'] * (quantity / 100), 1),
                'carbs': round(nutrition['carbs'] * (quantity / 100), 1),
                'fat': round(nutrition['fat'] * (quantity / 100), 1),
                'fiber': round(nutrition['fiber'] * (quantity / 100), 1)
            }
            
            # Log the meal
            meal_entry = {
                'timestamp': datetime.now().isoformat(),
                'food': pred_class,
                'quantity': quantity,
                'nutrition': adjusted_nutrition,
                'confidence': conf,
                'image_filename': filename
            }
            meal_history.append(meal_entry)
            
            # Keep only last 50 meals
            if len(meal_history) > 50:
                meal_history.pop(0)
                
            return jsonify({
                'success': True,
                'prediction': pred_class,
                'confidence': conf,
                'nutrition': adjusted_nutrition,
                'meal_id': len(meal_history) - 1
            })
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")  # Debug
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/meals', methods=['GET'])
def get_meals():
    return jsonify(meal_history)

@app.route('/meal/<int:meal_id>', methods=['DELETE'])
def delete_meal(meal_id):
    if 0 <= meal_id < len(meal_history):
        deleted_meal = meal_history.pop(meal_id)
        return jsonify({'success': True, 'deleted': deleted_meal})
    return jsonify({'error': 'Meal not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
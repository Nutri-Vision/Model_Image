# Nutri-Vision: Food Image Classifier with Nutrition Info

**Nutri-Vision** is a deep learning project that classifies images of food and provides their nutritional information. It uses **PyTorch** with a **ResNet-50 backbone**, and can be extended to include additional datasets like **Food-101** alongside custom fruit datasets.

---

## Features

- Classifies multiple food categories.
- Provides nutritional values (calories, protein, fat, carbs, etc.).
- Supports inference via a simple `main.py` script.
- Easy to extend with new datasets.
- Handles large model files using **Git LFS**.
- Lightweight training option for testing (small number of epochs/images).

---

## Project Structure

food_classifier/
├─ app.py # Flask app for web interface
├─ main.py # Script for predicting nutrition for an image
├─ train_food_classifier.py # Training script
├─ dataprep2.py # Optional data preparation script
├─ examples/ # Example images for testing
├─ model1/ # Contains trained model and class mapping
├─ runs/ # Training outputs and checkpoints
├─ templates/ # Flask HTML templates
├─ requirements.txt # Python dependencies
├─ .gitignore
├─ nutrition_data.json # Nutrition mapping for all food classes

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Nutri-Vision/Model_Nvision.git
cd food_classifier
Create and activate a virtual environment:
python3 -m venv v1
source v1/bin/activate  # macOS/Linux
v1\Scripts\activate     # Windows
Install dependencies:
pip install -r requirements.txt
(Optional) Install Git LFS for handling large .pth models:
git lfs install

## Inference
python main.py
Modify IMAGE_PATH in main.py to test different images.
Outputs class prediction and confidence.
Uses model1/best_model.pth and model1/class_mapping.json.


## Web App (Optional)
python app.py
Open http://127.0.0.1:5001 in your browser.
Upload an image to get predicted food class and nutritional info.


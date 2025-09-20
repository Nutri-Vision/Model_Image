import os
import random
import shutil
from pathlib import Path

# Original dataset path (after extracting Food-101)
original_dir = Path("food-101/images")
output_dir = Path("food-101-mini")

# Parameters
max_images_per_class = 100  # keep only 100 images per category
train_split = 0.8           # 80% train, 20% val

# Make folders
train_dir = output_dir / "train"
val_dir = output_dir / "val"
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Go through all categories
for class_folder in original_dir.iterdir():
    if class_folder.is_dir():
        images = list(class_folder.glob("*.jpg"))
        random.shuffle(images)

        # Pick only N images
        images = images[:max_images_per_class]

        # Split into train/val
        split_idx = int(len(images) * train_split)
        train_imgs, val_imgs = images[:split_idx], images[split_idx:]

        # Copy to train folder
        train_class_dir = train_dir / class_folder.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        for img in train_imgs:
            shutil.copy2(img, train_class_dir / img.name)

        # Copy to val folder
        val_class_dir = val_dir / class_folder.name
        val_class_dir.mkdir(parents=True, exist_ok=True)
        for img in val_imgs:
            shutil.copy2(img, val_class_dir / img.name)

print("✅ Mini Food-101 dataset created at:", output_dir)

"""
train_food_classifier.py

Usage examples:
  # Train on combined dataset
  python train_food_classifier.py --data_dirs ./fruits_data ./food101_mini --epochs 12 --batch_size 32 --lr 3e-4 --out_dir ./runs/exp1

  # Inference (after training):
  python train_food_classifier.py --predict_path ./examples/pizza.jpg --model_path ./runs/exp1/best_model.pth --predict

Requirements:
  - Python 3.8+
  - PyTorch 2.4+ (MPS support for Mac)
  - torchvision
  - Pillow
  - tqdm
"""

import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from PIL import Image

# ------------------ TRANSFORMS ------------------ #
def get_transforms(image_size=224):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size*1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_transforms, val_transforms

# ------------------ MODEL ------------------ #
def build_model(num_classes, pretrained=True, dropout=0.5):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model

# ------------------ TRAIN & VALIDATION ------------------ #
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ------------------ CHECKPOINTS ------------------ #
def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

# ------------------ PREDICTION ------------------ #
def predict_image(model, img_path, class_idx_to_name, device, image_size=224):
    transform = transforms.Compose([
        transforms.Resize(int(image_size*1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
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

# ------------------ DATA ------------------ #
def prepare_dataloaders(data_dirs, image_size=224, batch_size=32, num_workers=4):
    train_t, val_t = get_transforms(image_size=image_size)

    train_datasets, val_datasets = [], []
    class_to_idx = {}
    offset = 0

    for data_dir in data_dirs:
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")

        # Load dataset
        t_dataset = datasets.ImageFolder(train_dir, transform=train_t)
        v_dataset = datasets.ImageFolder(val_dir, transform=val_t)

        # Re-map class indices to avoid clashes
        t_dataset.targets = [y + offset for y in t_dataset.targets]
        v_dataset.targets = [y + offset for y in v_dataset.targets]

        # Merge class mapping
        for cls, idx in t_dataset.class_to_idx.items():
            class_to_idx[cls] = idx + offset

        offset += len(t_dataset.classes)

        train_datasets.append(t_dataset)
        val_datasets.append(v_dataset)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    classes = list(class_to_idx.keys())
    return train_loader, val_loader, classes

# ------------------ MAIN ------------------ #
def main(args):
    # Device detection: CUDA, MPS (Mac), or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Prepare data
    train_loader, val_loader, classes = prepare_dataloaders(
        args.data_dirs, image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    num_classes = len(classes)
    print(f"Found {num_classes} classes across datasets.")

    # Save class mapping
    idx_to_class = {i: c for i, c in enumerate(classes)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "class_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)

    # Build model
    model = build_model(num_classes, pretrained=not args.no_pretrained, dropout=args.dropout)
    model = model.to(device)

    # Optionally freeze backbone
    if args.freeze_backbone_epochs > 0:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("Backbone frozen (only classifier will train) for initial epochs.")

    # Criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Unfreeze backbone after freeze epochs
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.2, weight_decay=args.weight_decay)
            print("Backbone unfrozen. Fine-tuning entire model.")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        # Scheduler step
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = out_dir / "best_model.pth"
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_mapping": idx_to_class,
                "val_acc": val_acc
            }, str(save_path))
            print(f"Saved best model -> {save_path}")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = out_dir / f"ckpt_epoch{epoch}.pth"
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_mapping": idx_to_class,
                "val_acc": val_acc
            }, str(ckpt_path))
            print(f"Saved checkpoint -> {ckpt_path}")

    print("Training finished. Best val acc:", best_val_acc)

    # Prediction if requested
    if args.predict and args.predict_path and args.model_path:
        cp = load_checkpoint(args.model_path, device)
        model.load_state_dict(cp["model_state"])
        class_map = cp.get("class_mapping", idx_to_class)
        pred_class, conf = predict_image(model, args.predict_path, class_map, device, image_size=args.image_size)
        print(f"Prediction: {pred_class} (confidence: {conf:.3f})")

# ------------------ ARGUMENTS ------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, nargs="+", default=["./food_data"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="./runs/exp")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    main(args)

"""
Real Dataset Auto-Processing & EfficientNet-B0 Fine-Tuning Pipeline
Scans real images in 'Mango Dataset', extracts high-quality pathology crops,
organizes train/val splits, and fine-tunes the CNN classifier.
"""

import os
import sys
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Include backend in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from classifier import (
    DISEASE_CLASSES,
    CLASS_MAP,
    NAME_TO_META,
    build_efficientnet_classifier
)
from detector import YOLOv8Detector

MANGO_CLASSES = [d["name"] for d in DISEASE_CLASSES]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(MANGO_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(MANGO_CLASSES)}


def prepare_real_dataset_crops(raw_dataset_dir, output_dir, val_split=0.20):
    """
    Extracts pathology lesion crops from real mango leaf photos and builds train/val folders.
    """
    print(f"\n[Data Preparation] Scanning '{raw_dataset_dir}' for real images...")
    
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPEG", "*.JPG", "*.PNG"]:
        image_files.extend(glob.glob(os.path.join(raw_dataset_dir, ext)))
        image_files.extend(glob.glob(os.path.join(raw_dataset_dir, "**", ext), recursive=True))

    image_files = list(set(image_files))
    print(f"[Data Preparation] Found {len(image_files)} real mango leaf images.")

    if len(image_files) == 0:
        print("[Data Preparation] Warning: No images found.")
        return False

    detector = YOLOv8Detector()
    
    # Create output directories for all 8 classes
    for split in ["train", "val"]:
        for cls_name in MANGO_CLASSES:
            os.makedirs(os.path.join(output_dir, split, cls_name), exist_ok=True)

    random.seed(42)
    random.shuffle(image_files)

    total_crops = {cls_name: 0 for cls_name in MANGO_CLASSES}

    for idx, img_path in enumerate(image_files):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            np_rgb = np.array(pil_img)
            h, w = np_rgb.shape[:2]

            # Detect candidate lesion regions
            detections = detector.detect_regions(np_rgb)
            is_val = (idx < int(len(image_files) * val_split))
            split = "val" if is_val else "train"

            img_basename = os.path.splitext(os.path.basename(img_path))[0].replace(" ", "_")

            if not detections:
                # Whole leaf check (Healthy or diffuse foliage)
                save_dir = os.path.join(output_dir, split, "Healthy")
                out_path = os.path.join(save_dir, f"{img_basename}_full.jpg")
                pil_img.resize((256, 256)).save(out_path, quality=92)
                total_crops["Healthy"] += 1
            else:
                for c_idx, det in enumerate(detections):
                    x1, y1, x2, y2 = det["bbox"]
                    pad = 4
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(w, x2 + pad)
                    cy2 = min(h, y2 + pad)

                    crop_np = np_rgb[cy1:cy2, cx1:cx2]
                    if crop_np.shape[0] < 12 or crop_np.shape[1] < 12:
                        continue

                    crop_pil = Image.fromarray(crop_np)

                    # Determine botanical pathology category based on spectral chromaticity
                    r_mean = np.mean(crop_np[:, :, 0])
                    g_mean = np.mean(crop_np[:, :, 1])
                    b_mean = np.mean(crop_np[:, :, 2])
                    y_mean = 0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean

                    if y_mean > 175 and abs(r_mean - g_mean) < 20 and abs(g_mean - b_mean) < 20:
                        target_class = "Powdery Mildew"
                    elif y_mean < 55 and g_mean < 60:
                        target_class = "Sooty Mold"
                    elif r_mean > b_mean + 15 and r_mean > g_mean - 5 and (x1 < w * 0.15 or x2 > w * 0.85 or y1 < h * 0.15 or y2 > h * 0.85):
                        target_class = "Die Back"
                    elif r_mean > 120 and g_mean > 95 and b_mean < 110 and r_mean > b_mean + 15:
                        target_class = "Bacterial Canker"
                    elif r_mean > b_mean + 6 and r_mean > g_mean - 12:
                        target_class = "Anthracnose"
                    elif g_mean > r_mean * 1.15 and g_mean > b_mean * 1.2:
                        target_class = "Healthy"
                    else:
                        target_class = "Anthracnose"

                    save_dir = os.path.join(output_dir, split, target_class)
                    out_path = os.path.join(save_dir, f"{img_basename}_crop_{c_idx}.jpg")
                    crop_pil.resize((224, 224)).save(out_path, quality=92)
                    total_crops[target_class] += 1

        except Exception as e:
            continue

    print("\n[Data Preparation] Extracted Pathology Crop Statistics:")
    for cls_name, count in total_crops.items():
        print(f"  * {cls_name:20s}: {count:4d} crops")

    return True


class RealMangoCropDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []

        for cls_name in MANGO_CLASSES:
            cls_folder = os.path.join(self.split_dir, cls_name)
            if os.path.exists(cls_folder):
                for img_name in os.listdir(cls_folder):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        img_path = os.path.join(cls_folder, img_name)
                        self.samples.append((img_path, CLASS_TO_IDX[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(0, 128, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def fine_tune_efficientnet(crops_dir, models_dir, epochs=15, batch_size=16, lr=2e-4):
    """
    Fine-tunes EfficientNet-B0 with AdamW, Cosine Annealing, and data augmentations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Starting EfficientNet-B0 fine-tuning on device: {device}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RealMangoCropDataset(crops_dir, split="train", transform=train_transforms)
    val_dataset = RealMangoCropDataset(crops_dir, split="val", transform=val_transforms)

    if len(train_dataset) == 0:
        print("[Training] Error: No training samples available.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None

    print(f"[Training] Dataset Split -> Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    model = build_efficientnet_classifier(num_classes=len(MANGO_CLASSES), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0.0
    best_weights_path = os.path.join(models_dir, "mango_cnn_efficientnet.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = (correct / max(1, total)) * 100.0

        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            model.eval()
            val_running = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_running / max(1, val_total)
            val_acc = (val_correct / max(1, val_total)) * 100.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if train_acc >= best_acc:
            best_acc = train_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "accuracy": train_acc,
                "classes": MANGO_CLASSES
            }, best_weights_path)

    print(f"\n[Training] Fine-tuning Complete! Saved fine-tuned weights to: {best_weights_path} (Best Acc: {best_acc:.2f}%)")


if __name__ == "__main__":
    raw_dataset = os.path.join(os.path.dirname(BASE_DIR), "Mango Dataset")
    crops_dir = os.path.join(BASE_DIR, "data", "mango_crops")
    models_dir = os.path.join(BASE_DIR, "models")

    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    success = prepare_real_dataset_crops(raw_dataset, crops_dir)
    if success:
        fine_tune_efficientnet(crops_dir, models_dir, epochs=12, batch_size=16)

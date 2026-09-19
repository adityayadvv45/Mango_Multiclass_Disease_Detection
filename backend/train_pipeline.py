"""
PyTorch Multi-Model Training & Weight Export Pipeline.
Trains high-performance EfficientNet-B0 and MobileNetV3-Large classifiers
on exactly 300 images per single-disease class with a held-out test split.
"""

import os
import sys

# Configure UTF-8 stdout for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import Dict, Tuple, List

from backend.models import (
    CANONICAL_CLASSES,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    normalize_class_name
)

DATA_ROOT = os.path.join(PROJECT_ROOT, "backend", "data", "Mango S data")
BUNDLE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "backend", "models", "mango_model_bundle.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 6
SEED = 42
TRAIN_SAMPLES_PER_CLASS = 300

class CustomMangoDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def prepare_data_splits(data_root: str, train_per_class: int = TRAIN_SAMPLES_PER_CLASS, seed: int = SEED):
    """
    Scans the 8 single-disease classes, samples exactly 300 images per class for training,
    and reserves the rest for testing/validation with zero leakage.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 8 Single-Disease Classes (exclude multi-disease folder)
    single_disease_folders = [
        "Anthracnose",
        "Bacterial Canker",
        "Cutting Weevil",
        "Die Back",
        "Gall Midge",
        "Healthy",
        "Powdery Mildew",
        "Sooty Mould"
    ]
    
    # Map to canonical names
    classes = [normalize_class_name(f) for f in single_disease_folders]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    class_stats = {}
    
    for folder in single_disease_folders:
        folder_path = os.path.join(data_root, folder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG')
        all_imgs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(valid_exts)]
        
        # Sort first for reproducibility, then shuffle with seed
        all_imgs = sorted(all_imgs)
        random.shuffle(all_imgs)
        
        canonical_name = normalize_class_name(folder)
        c_idx = class_to_idx[canonical_name]
        
        if len(all_imgs) < train_per_class:
            raise ValueError(f"Class '{folder}' has only {len(all_imgs)} images, expected at least {train_per_class}.")
            
        selected_train = all_imgs[:train_per_class]
        selected_val = all_imgs[train_per_class:]
        
        train_paths.extend(selected_train)
        train_labels.extend([c_idx] * len(selected_train))
        
        val_paths.extend(selected_val)
        val_labels.extend([c_idx] * len(selected_val))
        
        class_stats[canonical_name] = {
            "total": len(all_imgs),
            "train": len(selected_train),
            "val": len(selected_val)
        }
        
    return (train_paths, train_labels), (val_paths, val_labels), classes, class_stats

def train_and_export():
    os.makedirs(os.path.dirname(BUNDLE_OUTPUT_PATH), exist_ok=True)
    
    torch.manual_seed(SEED)
    print("=" * 60)
    print("MANGO LEAF MODEL TRAINING & INTEGRATION PIPELINE")
    print("=" * 60)
    print(f"[TRAIN] Device: {DEVICE}")
    print(f"[TRAIN] Dataset root: {DATA_ROOT}")
    print(f"[TRAIN] Training Images Per Class: {TRAIN_SAMPLES_PER_CLASS}")
    
    (train_paths, train_labels), (val_paths, val_labels), classes, stats = prepare_data_splits(DATA_ROOT)
    
    print("\n--- Dataset Distribution Summary ---")
    for c_name, s in stats.items():
        print(f"  {c_name:18s} -> Total: {s['total']:3d} | Train: {s['train']:3d} | Test/Val: {s['val']:3d}")
    print(f"  TOTALS: Train = {len(train_paths)} images, Test/Val = {len(val_paths)} images")
    print(f"  Classes: {classes}\n")
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    train_ds = CustomMangoDataset(train_paths, train_labels, transform=train_transform)
    val_ds = CustomMangoDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model_architectures = ["EfficientNet-B0", "MobileNetV3-Large"]
    trained_weights_dict = {}
    metrics_summary = {}
    
    for arch_name in model_architectures:
        print("=" * 60)
        print(f"[TRAIN] Training {arch_name} ({EPOCHS} epochs, Batch Size: {BATCH_SIZE})...")
        print("=" * 60)
        
        model = build_model(arch_name, num_classes=len(classes)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
        
        best_acc = 0.0
        best_wts = copy.deepcopy(model.state_dict())
        best_epoch_metrics = {}
        
        start_time = time.time()
        for epoch in range(EPOCHS):
            # Training Phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                train_loss += loss.item() * inputs.size(0)
                train_correct += torch.sum(preds == labels.data).item()
                
            scheduler.step()
            train_epoch_loss = train_loss / len(train_ds)
            train_epoch_acc = train_correct / len(train_ds)
            
            # Validation / Test Phase on unseen data
            model.eval()
            val_loss = 0.0
            val_correct = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * inputs.size(0)
                    val_correct += torch.sum(preds == labels.data).item()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
            val_epoch_loss = val_loss / len(val_ds)
            val_epoch_acc = val_correct / len(val_ds)
            
            p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            print(f"Epoch {epoch+1:2d}/{EPOCHS:2d} -> Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.2%} | Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.2%} | Macro F1: {f1:.4f}")
            
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                
                # Compute per-class F1-scores
                per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
                per_class_metrics = {classes[i]: float(per_class_f1[i]) for i in range(len(classes))}
                
                best_epoch_metrics = {
                    "Accuracy": float(val_epoch_acc),
                    "Precision": float(p),
                    "Recall": float(r),
                    "F1-Score": float(f1),
                    "Epoch": epoch + 1,
                    "PerClassF1": per_class_metrics,
                    "ConfusionMatrix": confusion_matrix(all_labels, all_preds).tolist()
                }
                
        elapsed = time.time() - start_time
        print(f"\n[TRAIN] {arch_name} complete in {elapsed:.1f}s. Best Unseen Test Accuracy: {best_acc:.2%}")
        print(f"Per-Class F1 Scores on Unseen Test Data:")
        for c, f in best_epoch_metrics["PerClassF1"].items():
            print(f"  {c:18s}: {f:.2%}")
            
        trained_weights_dict[arch_name] = best_wts
        metrics_summary[arch_name] = best_epoch_metrics
        
    print(f"\n[TRAIN] Saving final model bundle to {BUNDLE_OUTPUT_PATH}...")
    bundle = {
        "models": trained_weights_dict,
        "class_names": classes,
        "class_mapping": {i: c for i, c in enumerate(classes)},
        "metrics": metrics_summary,
        "train_samples_per_class": TRAIN_SAMPLES_PER_CLASS,
        "timestamp": time.time(),
        "img_size": IMG_SIZE,
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        }
    }
    torch.save(bundle, BUNDLE_OUTPUT_PATH)
    print("✅ Model bundle successfully saved!")
    return bundle

if __name__ == "__main__":
    train_and_export()

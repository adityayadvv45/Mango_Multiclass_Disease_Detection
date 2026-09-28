"""
PyTorch Multi-Model Training & Weight Export Pipeline.
Trains high-performance EfficientNet-B0 and MobileNetV3-Large classifiers
on the combined verified Mango Leaf dataset + Roboflow ground-truth training set.
"""

import os
import sys
import csv
import glob
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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Tuple, List

# Configure UTF-8 stdout for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.models import (
    CANONICAL_CLASSES,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    normalize_class_name
)

DATA_ROOT = os.path.join(PROJECT_ROOT, "backend", "data", "Mango S data")
ROBOFLOW_TRAIN_DIR = os.path.join(PROJECT_ROOT, "backend", "data", "Multiclass_leaf.v5i.yolov8-data", "train")
BUNDLE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "backend", "models", "mango_model_bundle.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 6
SEED = 42
TRAIN_SAMPLES_PER_CLASS = 300

ROBOFLOW_TO_CANONICAL = {
    0: "Anthracnose",
    1: "Bacterial Canker",
    2: "Powdery Mildew",
    3: "Die Back",
    4: "Gall Midge"
}

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

def load_roboflow_train_data(class_to_idx: Dict[str, int]) -> Tuple[List[str], List[int]]:
    """Loads images from Roboflow train split with their ground truth class labels."""
    images_dir = os.path.join(ROBOFLOW_TRAIN_DIR, "images")
    labels_dir = os.path.join(ROBOFLOW_TRAIN_DIR, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"[WARN] Roboflow train directory not found at {ROBOFLOW_TRAIN_DIR}")
        return [], []
        
    img_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png")))
    rf_paths, rf_labels = [], []
    
    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        classes_present = set()
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cid = int(parts[0])
                            if cid in ROBOFLOW_TO_CANONICAL:
                                classes_present.add(ROBOFLOW_TO_CANONICAL[cid])
                        except ValueError:
                            pass
                            
        if not classes_present:
            # Unlabeled in Roboflow = Healthy leaf
            c_name = "Healthy"
            if c_name in class_to_idx:
                rf_paths.append(img_path)
                rf_labels.append(class_to_idx[c_name])
        else:
            # For each disease present in this training image, add sample
            for c_name in classes_present:
                if c_name in class_to_idx:
                    rf_paths.append(img_path)
                    rf_labels.append(class_to_idx[c_name])
                    
    print(f"  [ROBOFLOW] Loaded {len(rf_paths)} training samples from {len(img_files)} Roboflow train images.")
    return rf_paths, rf_labels

def prepare_data_splits(data_root: str, train_per_class: int = TRAIN_SAMPLES_PER_CLASS, seed: int = SEED):
    """
    Scans the 8 single-disease classes, samples 300 images per class for training,
    reserves test images, and adds Roboflow train images to the training split.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    classes = CANONICAL_CLASSES
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    class_stats = {}
    
    for c_name in classes:
        folder_candidates = [c_name, "Sooty Mould" if c_name == "Sooty Mold" else c_name]
        folder_path = None
        for cand in folder_candidates:
            p = os.path.join(data_root, cand)
            if os.path.exists(p):
                folder_path = p
                break
                
        if not folder_path:
            raise FileNotFoundError(f"Folder not found for class '{c_name}' at {data_root}")
            
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG')
        all_imgs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(valid_exts)]
        all_imgs = sorted(all_imgs)
        random.shuffle(all_imgs)
        
        c_idx = class_to_idx[c_name]
        selected_train = all_imgs[:train_per_class]
        selected_val = all_imgs[train_per_class:]
        
        train_paths.extend(selected_train)
        train_labels.extend([c_idx] * len(selected_train))
        
        val_paths.extend(selected_val)
        val_labels.extend([c_idx] * len(selected_val))
        
        class_stats[c_name] = {
            "total": len(all_imgs),
            "train": len(selected_train),
            "val": len(selected_val)
        }
        
    # Augment training set with Roboflow train split
    rf_train_paths, rf_train_labels = load_roboflow_train_data(class_to_idx)
    train_paths.extend(rf_train_paths)
    train_labels.extend(rf_train_labels)
    
    return (train_paths, train_labels), (val_paths, val_labels), classes, class_stats

def save_history_to_csv(history: List[Dict], csv_path: str):
    """Saves training history using built-in csv module (no pandas dependency)."""
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"📄 Saved training history log -> '{csv_path}'")

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
    print(f"  TOTALS: Train = {len(train_paths)} samples, Test/Val = {len(val_paths)} images")
    print(f"  Classes: {classes}\n")
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
        history = []
        
        start_time = time.time()
        for epoch in range(EPOCHS):
            # Training Phase
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_train += torch.sum(preds == lbls.data).item()
                total_train += imgs.size(0)
                
            scheduler.step()
            train_loss = running_loss / total_train
            train_acc = correct_train / total_train
            
            # Validation Phase
            model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                    val_loss += loss.item() * imgs.size(0)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(lbls.cpu().numpy())
                    
            val_loss = val_loss / len(val_ds)
            val_acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
            
            p_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
            r_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
            f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "precision": round(p_macro, 4),
                "recall": round(r_macro, 4),
                "f1": round(f1_macro, 4),
            }
            history.append(epoch_log)
            print(f"  Epoch [{epoch+1:02d}/{EPOCHS:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | F1: {f1_macro:.4f}")
            
            if val_acc >= best_acc:
                best_acc = val_acc
                best_wts = copy.deepcopy(model.state_dict())
                best_epoch_metrics = epoch_log
                
        elapsed = time.time() - start_time
        print(f"  [DONE] {arch_name} trained in {elapsed:.1f}s. Best Val Accuracy: {best_acc:.2%}")
        
        # Save training history CSV
        csv_filename = os.path.join(PROJECT_ROOT, "backend", "models", f"{arch_name.lower()}_training_log.csv")
        save_history_to_csv(history, csv_filename)
        
        trained_weights_dict[arch_name] = best_wts
        metrics_summary[arch_name] = {
            "Accuracy": round(best_epoch_metrics.get("val_acc", best_acc), 4),
            "Precision": round(best_epoch_metrics.get("precision", 0.0), 4),
            "Recall": round(best_epoch_metrics.get("recall", 0.0), 4),
            "F1": round(best_epoch_metrics.get("f1", 0.0), 4),
            "FinalLoss": round(best_epoch_metrics.get("val_loss", 0.0), 4)
        }
        
    bundle = {
        "models": trained_weights_dict,
        "class_names": classes,
        "metrics": metrics_summary,
        "img_size": IMG_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    torch.save(bundle, BUNDLE_OUTPUT_PATH)
    print(f"\n✅ Successfully saved multi-model bundle to '{BUNDLE_OUTPUT_PATH}'")
    print(f"📊 Final Metrics Summary: {metrics_summary}\n")
    return metrics_summary

if __name__ == "__main__":
    train_and_export()

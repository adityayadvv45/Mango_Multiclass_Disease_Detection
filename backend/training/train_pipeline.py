"""
PyTorch Multi-Model Training & Weight Export Pipeline.
Trains high-performance, class-balanced EfficientNet-B0 and MobileNetV3-Large classifiers
strictly on the verified YOLO Multiclass Leaf dataset with class-imbalance mitigation.
"""

import os
import sys
import csv
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
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, hamming_loss
from typing import Dict, Tuple, List, Any

# Configure UTF-8 stdout for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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

# STRICT DATASET CONFIGURATION: Only use verified YOLO dataset; exclude Kaggle and older datasets
DATA_ROOT = os.path.join(PROJECT_ROOT, "backend", "data", "Multiclass_leaf.v5i.yolov8-data")
BUNDLE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "backend", "models", "mango_model_bundle.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 6
SEED = 42

YOLO_CLASSES_MAP = {
    0: "Anthracnose",
    1: "Bacterial Canker",
    2: "Powdery Mildew",
    3: "Die Back",
    4: "Gall Midge"
}

class CachedYoloMultiLabelDataset(Dataset):
    """
    High-speed in-memory dataset that pre-caches resized PIL images from the
    verified YOLO dataset, mapping annotations to multi-label target vectors.
    """
    def __init__(self, data_root: str, split: str = "train", transform=None):
        self.transform = transform
        self.classes = CANONICAL_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        img_dir = os.path.join(data_root, split, "images")
        lbl_dir = os.path.join(data_root, split, "labels")
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            raise FileNotFoundError(f"Missing images or labels directory for split '{split}' in {data_root}")
            
        self.cached_images = []
        self.targets = []
        self.image_names = []
        self.class_presence_counts = np.zeros(len(self.classes), dtype=np.int32)
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG')
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(valid_exts)])
        
        t0 = time.time()
        print(f"  [CACHE] Pre-loading & caching {len(img_files)} images for {split} split...", flush=True)
        for f in img_files:
            img_path = os.path.join(img_dir, f)
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
                self.cached_images.append(img_rgb)
            self.image_names.append(f)
            
            base, _ = os.path.splitext(f)
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            
            classes_in_img = set()
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as lf:
                    lines = [l.strip() for l in lf.readlines() if l.strip()]
                for l in lines:
                    try:
                        cid = int(l.split()[0])
                        cname = YOLO_CLASSES_MAP.get(cid)
                        if cname:
                            classes_in_img.add(cname)
                    except Exception:
                        continue
                        
            # Empty label files in YOLO format correspond to Healthy leaves
            if not classes_in_img:
                classes_in_img.add("Healthy")
                
            target_vec = np.zeros(len(self.classes), dtype=np.float32)
            for cname in classes_in_img:
                idx = self.class_to_idx[cname]
                target_vec[idx] = 1.0
                self.class_presence_counts[idx] += 1
                
            self.targets.append(target_vec)
            
        print(f"  [CACHE] Successfully cached {len(self.cached_images)} images in {time.time()-t0:.2f}s.", flush=True)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.cached_images[idx]
        if self.transform:
            img = self.transform(img)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img, target

def save_history_to_csv(history: List[Dict], csv_path: str):
    """Saves training history log."""
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"📄 Saved training history log -> '{csv_path}'", flush=True)

def train_and_export():
    os.makedirs(os.path.dirname(BUNDLE_OUTPUT_PATH), exist_ok=True)
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("=" * 75, flush=True)
    print("🚀 MANGO LEAF MULTI-MODEL BALANCED TRAINING PIPELINE (YOLO-DATASET ONLY)", flush=True)
    print("=" * 75, flush=True)
    print(f"[TRAIN] Device: {DEVICE}", flush=True)
    print(f"[TRAIN] Primary & Only Dataset: {DATA_ROOT}", flush=True)
    print(f"[TRAIN] Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}", flush=True)
    
    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    train_ds = CachedYoloMultiLabelDataset(DATA_ROOT, split="train", transform=train_transform)
    val_ds = CachedYoloMultiLabelDataset(DATA_ROOT, split="valid", transform=val_transform)
    test_ds = CachedYoloMultiLabelDataset(DATA_ROOT, split="test", transform=val_transform)
    
    print("\n--- Training Set Class Distribution ---", flush=True)
    for idx, cname in enumerate(CANONICAL_CLASSES):
        count = train_ds.class_presence_counts[idx]
        pct = (count / len(train_ds)) * 100
        print(f"  {cname:18s} -> {count:3d} images ({pct:.1f}%)", flush=True)
        
    print(f"\n  TOTALS: Train = {len(train_ds)}, Validation = {len(val_ds)}, Test = {len(test_ds)} images", flush=True)
    
    # Compute inverse class positive weights to counteract Anthracnose frequency dominance
    pos_weights = []
    n_train = len(train_ds)
    for count in train_ds.class_presence_counts:
        if count > 0:
            w = (n_train - count) / float(count)
        else:
            w = 1.0
        pos_weights.append(min(w, 20.0))
        
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)
    print(f"  Class Balancing Loss pos_weights: {[round(w, 3) for w in pos_weights]}", flush=True)
    print(f"  Canonical Target Classes ({len(CANONICAL_CLASSES)}): {CANONICAL_CLASSES}\n", flush=True)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model_architectures = ["EfficientNet-B0", "MobileNetV3-Large"]
    trained_weights_dict = {}
    metrics_summary = {}
    
    for arch_name in model_architectures:
        print("=" * 75, flush=True)
        print(f"[TRAIN] Training {arch_name} ({EPOCHS} epochs, Batch Size: {BATCH_SIZE})...", flush=True)
        print("=" * 75, flush=True)
        
        model = build_model(arch_name, num_classes=len(CANONICAL_CLASSES)).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
        
        best_f1 = 0.0
        best_epoch = 1
        best_wts = copy.deepcopy(model.state_dict())
        best_epoch_metrics = {}
        history = []
        
        start_time = time.time()
        for epoch in range(EPOCHS):
            # Training Phase
            model.train()
            running_loss = 0.0
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                
            scheduler.step()
            train_loss = running_loss / len(train_ds)
            
            # Validation Phase
            model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * imgs.size(0)
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).cpu().numpy().astype(int)
                    all_preds.append(preds)
                    all_targets.append(targets.cpu().numpy().astype(int))
                    
            val_loss = val_loss / len(val_ds)
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            
            active_mask = np.sum(all_targets, axis=0) > 0
            f1_macro = f1_score(all_targets[:, active_mask], all_preds[:, active_mask], average='macro', zero_division=0)
            p_macro = precision_score(all_targets[:, active_mask], all_preds[:, active_mask], average='macro', zero_division=0)
            r_macro = recall_score(all_targets[:, active_mask], all_preds[:, active_mask], average='macro', zero_division=0)
            h_loss = hamming_loss(all_targets, all_preds)
            
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "precision": round(p_macro, 4),
                "recall": round(r_macro, 4),
                "f1": round(f1_macro, 4),
                "hamming_loss": round(h_loss, 4)
            }
            history.append(epoch_log)
            print(f"  Epoch [{epoch+1:02d}/{EPOCHS:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {f1_macro:.4f} | Prec: {p_macro:.4f} | Rec: {r_macro:.4f} | HammingLoss: {h_loss:.4f}", flush=True)
            
            if f1_macro >= best_f1:
                best_f1 = f1_macro
                best_epoch = epoch + 1
                best_wts = copy.deepcopy(model.state_dict())
                best_epoch_metrics = epoch_log
                
        elapsed = time.time() - start_time
        print(f"  [DONE] {arch_name} trained in {elapsed:.1f}s. Best Epoch: {best_epoch} (Val Macro F1: {best_f1:.4f})", flush=True)
        
        # Test evaluation with best checkpoint
        model.load_state_dict(best_wts)
        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                outputs = model(imgs)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).cpu().numpy().astype(int)
                test_preds.append(preds)
                test_targets.append(targets.cpu().numpy().astype(int))
                
        test_preds = np.vstack(test_preds)
        test_targets = np.vstack(test_targets)
        
        active_test_mask = np.sum(test_targets, axis=0) > 0
        active_classes = [c for i, c in enumerate(CANONICAL_CLASSES) if active_test_mask[i]]
        
        print(f"\n📊 Detailed Test Evaluation Report for {arch_name}:", flush=True)
        report_dict = classification_report(test_targets[:, active_test_mask], test_preds[:, active_test_mask], target_names=active_classes, output_dict=True, zero_division=0)
        report_str = classification_report(test_targets[:, active_test_mask], test_preds[:, active_test_mask], target_names=active_classes, zero_division=0)
        print(report_str, flush=True)
        
        # Save training history CSV
        csv_filename = os.path.join(PROJECT_ROOT, "backend", "models", f"{arch_name.lower()}_training_log.csv")
        save_history_to_csv(history, csv_filename)
        
        trained_weights_dict[arch_name] = best_wts
        metrics_summary[arch_name] = {
            "BestValF1": round(best_f1, 4),
            "BestValPrecision": round(best_epoch_metrics.get("precision", 0.0), 4),
            "BestValRecall": round(best_epoch_metrics.get("recall", 0.0), 4),
            "TestPerClass": {
                c: {
                    "precision": round(report_dict.get(c, {}).get("precision", 0.0), 4),
                    "recall": round(report_dict.get(c, {}).get("recall", 0.0), 4),
                    "f1": round(report_dict.get(c, {}).get("f1-score", 0.0), 4),
                    "support": report_dict.get(c, {}).get("support", 0)
                } for c in active_classes
            }
        }
        
    bundle = {
        "models": trained_weights_dict,
        "class_names": CANONICAL_CLASSES,
        "metrics": metrics_summary,
        "img_size": IMG_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    torch.save(bundle, BUNDLE_OUTPUT_PATH)
    print(f"\n✅ Successfully saved multi-model bundle to '{BUNDLE_OUTPUT_PATH}'", flush=True)
    print(f"📊 Final Metrics Summary: {metrics_summary}\n", flush=True)
    return metrics_summary

if __name__ == "__main__":
    train_and_export()

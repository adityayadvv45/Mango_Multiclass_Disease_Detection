"""
====================================================================================================
RESEARCH PAPER ARTIFACT: DEEP LEARNING ARCHITECTURES & TRAINING PIPELINES
Project: Real-Time Mango Leaf Single & Multi-Pathology Disease Detection & Lesion Localization
Author / Study: Mango Leaf Foliar Diagnostic Research
====================================================================================================

This file provides publication-ready reference code structured for academic research papers:
- SECTION 1: Deep Convolutional Neural Network (CNN) Architectures (EfficientNet-B0, MobileNetV3, Custom CNN)
- SECTION 2: CNN Training & Validation Pipeline (Cosine Annealing, AdamW, Per-Class F1 Evaluation)
- SECTION 3: YOLOv8 Lesion Detection & Localization Pipeline (Training & Inference)
- SECTION 4: Foliar Leaf Blade Segmentation & Chromatic Background Rejection Engine
====================================================================================================
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Optional


# ==================================================================================================
# SECTION 1: DEEP CNN MODEL ARCHITECTURES
# ==================================================================================================

class CustomResidualMangoCNN(nn.Module):
    """
    Lightweight Custom Residual CNN with Squeeze-and-Excitation (SE) Attention Blocks
    designed for low-latency foliar disease classification.
    """
    def __init__(self, num_classes: int = 8, in_channels: int = 3, dropout_rate: float = 0.3):
        super(CustomResidualMangoCNN, self).__init__()
        
        # Initial Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Residual Stage 1 (64 -> 128)
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.shortcut1 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        
        # Residual Stage 2 (128 -> 256)
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.shortcut2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        
        # Global Pooling & Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.act(self.stage1(x) + self.shortcut1(x))
        x = self.act(self.stage2(x) + self.shortcut2(x))
        x = self.gap(x)
        out = self.classifier(x)
        return out


def build_cnn_model(arch_name: str = "EfficientNet-B0", num_classes: int = 8) -> nn.Module:
    """
    Factory builder for Deep Transfer Learning and Custom CNN Backbones.
    Supports: 'EfficientNet-B0', 'MobileNetV3-Large', 'Custom-ResCNN', 'ResNet50'.
    """
    if arch_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    elif arch_name == "MobileNetV3-Large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif arch_name == "Custom-ResCNN":
        model = CustomResidualMangoCNN(num_classes=num_classes)
    elif arch_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model backbone: {arch_name}")
        
    return model


# ==================================================================================================
# SECTION 2: CNN TRAINING, OPTIMIZATION & PER-CLASS VALIDATION PIPELINE
# ==================================================================================================

class MangoFoliarDataset(Dataset):
    """Custom PyTorch Dataset for Mango Leaf Disease Classification."""
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def train_cnn_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    arch_name: str = "EfficientNet-B0",
    num_classes: int = 8,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """
    Executes training with AdamW, Cosine Annealing learning rate schedule,
    and comprehensive evaluation on unseen validation sets.
    """
    model = build_cnn_model(arch_name, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_accuracy = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "macro_f1": []}

    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        scheduler.step()
        train_epoch_loss = running_loss / total_train
        train_epoch_acc = correct_train / total_train
        
        # 2. Validation Phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_epoch_loss = val_loss / total_val
        val_epoch_acc = correct_val / total_val
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        
        history["train_loss"].append(train_epoch_loss)
        history["train_acc"].append(train_epoch_acc)
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc)
        history["macro_f1"].append(macro_f1)
        
        print(f"Epoch [{epoch+1:2d}/{epochs:2d}] -> Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.2%} | Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.2%} | F1: {macro_f1:.4f}")
        
    return {"model": model, "history": history}


# ==================================================================================================
# SECTION 3: YOLOV8 LESION LOCALIZATION & MULTI-DISEASE DETECTION
# ==================================================================================================

def train_yolov8_detector(
    data_yaml_path: str = "mango_data.yaml",
    model_size: str = "yolov8n.pt",
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16
):
    """
    Fine-tunes an Ultralytics YOLOv8 object detector for mango lesion localization.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[WARN] Ultralytics package required for YOLOv8 execution (`pip install ultralytics`).")
        return None

    model = YOLO(model_size)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        patience=10,
        project="MangoLesionDetection",
        name="yolov8_lesion_experiment"
    )
    metrics = model.val()
    return model, metrics


def infer_yolov8_lesions(model, image_bgr: np.ndarray, conf_threshold: float = 0.35) -> List[Dict]:
    """
    Runs YOLOv8 forward inference and extracts bounding box predictions.
    """
    results = model.predict(source=image_bgr, conf=conf_threshold, iou=0.45, verbose=False)
    detections = []
    
    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = model.names[cls_id]
            
            detections.append({
                "class_name": cls_name,
                "confidence": round(conf * 100, 2),
                "box": coords.tolist() # [xmin, ymin, xmax, ymax]
            })
    return detections


# ==================================================================================================
# SECTION 4: FOLIAR LEAF BLADE SEGMENTATION & BACKGROUND REJECTION ENGINE
# ==================================================================================================

def segment_mango_leaf_blade(image_bgr: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
    """
    Multi-Cue Foliar Segmentation Engine:
    Accurately isolates the continuous mango leaf blade and strictly rejects non-foliar
    artifacts (white paper, human skin/hands, wooden tables, soil, and wall textures).
    """
    h, w = image_bgr.shape[:2]
    img_area = h * w
    
    # 1. Human Skin Tone Rejection (Combined YCrCb + HSV)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    skin_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 175, 127]))
    skin_hsv = cv2.inRange(hsv, np.array([0, 40, 60]), np.array([25, 200, 255]))
    skin_mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)
    
    # 2. White Paper / Notebook Sheet Rejection
    paper_mask1 = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 45, 255]))
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    neutral_white = (np.abs(r - g) < 30) & (np.abs(g - b) < 30) & ((r + g + b) / 3.0 > 165)
    paper_mask = cv2.bitwise_or(paper_mask1, (neutral_white.astype(np.uint8)) * 255)
    
    # 3. Excess Green Index (ExG = 2G - R - B) for Foliar Extraction
    exg = 2.0 * g - r - b
    foliage_green = cv2.inRange(hsv, np.array([20, 20, 20]), np.array([98, 255, 255]))
    necrosis_hsv = cv2.inRange(hsv, np.array([5, 30, 20]), np.array([25, 255, 240]))
    powdery_hsv = cv2.inRange(hsv, np.array([15, 10, 100]), np.array([105, 100, 255]))
    
    candidate_leaf = cv2.bitwise_or(foliage_green, necrosis_hsv)
    candidate_leaf = cv2.bitwise_or(candidate_leaf, powdery_hsv)
    candidate_leaf = cv2.bitwise_or(candidate_leaf, (exg > -5).astype(np.uint8) * 255)
    
    # 4. Otsu Inverse for Light/Paper Backgrounds
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    corner_pixels = np.vstack([
        image_bgr[:15, :15].reshape(-1, 3), image_bgr[:15, -15:].reshape(-1, 3),
        image_bgr[-15:, :15].reshape(-1, 3), image_bgr[-15:, -15:].reshape(-1, 3)
    ])
    corner_mean = float(np.mean(corner_pixels))
    
    if corner_mean > 140:
        candidate_leaf = cv2.bitwise_or(candidate_leaf, otsu_dark)
        candidate_leaf = cv2.bitwise_and(candidate_leaf, cv2.bitwise_not(paper_mask))
        candidate_leaf = cv2.bitwise_and(candidate_leaf, cv2.bitwise_not(skin_mask))
    else:
        soil_mask = ((r > (g + 10)) & (exg < -15) & (r > b)).astype(np.uint8) * 255
        candidate_leaf = cv2.bitwise_and(candidate_leaf, cv2.bitwise_not(paper_mask))
        candidate_leaf = cv2.bitwise_and(candidate_leaf, cv2.bitwise_not(skin_mask))
        candidate_leaf = cv2.bitwise_and(candidate_leaf, cv2.bitwise_not(soil_mask))
        
    # Morphological cleanup & contour filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_leaf = cv2.morphologyEx(candidate_leaf, cv2.MORPH_OPEN, kernel, iterations=2)
    candidate_leaf = cv2.morphologyEx(candidate_leaf, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(candidate_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8), False, {"reason": "No valid leaf contour"}
        
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 0.005 * img_area:
        return np.zeros((h, w), dtype=np.uint8), False, {"reason": "Contour area below threshold"}
        
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(leaf_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    leaf_pixels = int(cv2.countNonZero(leaf_mask))
    leaf_ratio = leaf_pixels / float(img_area)
    
    if leaf_ratio < 0.007:
        return np.zeros((h, w), dtype=np.uint8), False, {"reason": "Foliar coverage ratio insufficient"}
        
    return leaf_mask, True, {"leaf_area_ratio": leaf_ratio}


def is_bbox_inside_foliar_mask(bbox: Tuple[int, int, int, int], leaf_mask: np.ndarray, min_overlap: float = 0.45) -> bool:
    """
    Validates whether a detected bounding box [ymin, xmin, ymax, xmax] is genuinely located
    on the leaf blade and rejects any detections on external backgrounds.
    """
    ymin, xmin, ymax, xmax = bbox
    h, w = leaf_mask.shape[:2]
    ymin, ymax = max(0, min(ymin, h - 1)), max(0, min(ymax, h))
    xmin, xmax = max(0, min(xmin, w - 1)), max(0, min(xmax, w))
    
    area = (xmax - xmin) * (ymax - ymin)
    if area <= 0:
        return False
        
    leaf_pixels = cv2.countNonZero(leaf_mask[ymin:ymax, xmin:xmax])
    overlap_ratio = leaf_pixels / float(area)
    center_y, center_x = (ymin + ymax) // 2, (xmin + xmax) // 2
    
    return (overlap_ratio >= min_overlap) and (leaf_mask[center_y, center_x] > 0)


# ==================================================================================================
# SCRIPT EXECUTION DEMONSTRATION
# ==================================================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("MANGO LEAF DISEASE DIAGNOSTIC MODEL SUITE (FOR RESEARCH PUBLICATION)")
    print("=" * 80)
    
    # 1. Instantiate CNN Architecture
    model_eff = build_cnn_model("EfficientNet-B0", num_classes=8)
    total_params = sum(p.numel() for p in model_eff.parameters() if p.requires_grad)
    print(f"[OK] EfficientNet-B0 Initialized (Trainable Parameters: {total_params:,})")
    
    # 2. Instantiate Custom ResCNN
    model_custom = build_cnn_model("Custom-ResCNN", num_classes=8)
    custom_params = sum(p.numel() for p in model_custom.parameters() if p.requires_grad)
    print(f"[OK] Custom-ResCNN Initialized (Trainable Parameters: {custom_params:,})")
    
    # 3. Simulate Forward Pass with Dummy Tensor (Batch=4, Channels=3, H=224, W=224)
    dummy_input = torch.randn(4, 3, 224, 224)
    out_eff = model_eff(dummy_input)
    out_custom = model_custom(dummy_input)
    print(f"[OK] Forward Pass Successful: EfficientNet Output Shape = {out_eff.shape}")
    print(f"[OK] Forward Pass Successful: Custom-ResCNN Output Shape = {out_custom.shape}")
    print("=" * 80)

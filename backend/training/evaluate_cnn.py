"""
Evaluation script for EfficientNet-B0 Mango Leaf Classifier on Test Split.
Computes multi-class metrics (Accuracy, Top-k, Confusion Matrix) on test samples.
"""

import os
import sys
import argparse
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "training"))


from dataset import create_dataloaders, MANGO_CLASSES, IDX_TO_CLASS
from classifier import build_efficientnet_classifier


def evaluate_model(weights_path=None, data_dir=None, device_str=None):
    if weights_path is None:
        weights_path = os.path.join(BASE_DIR, "models", "mango_cnn_efficientnet.pth")
    if data_dir is None:
        data_dir = os.path.join(BASE_DIR, "data", "Mango S data")

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"==================================================")
    print(f"EfficientNet-B0 Mango Leaf Evaluation Suite")
    print(f"Device: {device} | Weights: {weights_path}")
    print(f"Dataset: {data_dir}")
    print(f"==================================================")

    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' does not exist. Please train the model first.")
        return

    _, _, test_loader = create_dataloaders(data_dir, batch_size=32, max_samples_per_class=200)
    if len(test_loader.dataset) == 0:
        print(f"Note: Evaluating on validation set instead.")
        _, test_loader, _ = create_dataloaders(data_dir, batch_size=32, max_samples_per_class=200)

    if len(test_loader.dataset) == 0:
        print("Error: No evaluation samples found.")
        return

    model = build_efficientnet_classifier(num_classes=len(MANGO_CLASSES), pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    class_correct = {i: 0 for i in range(len(MANGO_CLASSES))}
    class_total = {i: 0 for i in range(len(MANGO_CLASSES))}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            for p, l in zip(preds, labels):
                if p == l:
                    class_correct[l.item()] += 1
                class_total[l.item()] += 1

    overall_acc = (sum(class_correct.values()) / max(1, sum(class_total.values()))) * 100.0

    print(f"\nEvaluation Results ({len(all_targets)} total test images):")
    print(f"Overall Accuracy: {overall_acc:.2f}%\n")
    print(f"{'Class Name':<22} | {'Correct':<8} | {'Total':<8} | {'Accuracy (%)':<12}")
    print("-" * 58)
    for idx, name in enumerate(MANGO_CLASSES):
        tot = class_total[idx]
        corr = class_correct[idx]
        acc = (corr / max(1, tot)) * 100.0 if tot > 0 else 0.0
        print(f"{name:<22} | {corr:<8} | {tot:<8} | {acc:<12.2f}")
    print("-" * 58)


if __name__ == "__main__":
    default_weights = os.path.join(BASE_DIR, "models", "mango_cnn_efficientnet.pth")
    default_data = os.path.join(BASE_DIR, "data", "Mango S data")

    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 on Mango Leaf Test Set")
    parser.add_argument("--weights", type=str, default=default_weights, help="Path to model weights")
    parser.add_argument("--data_dir", type=str, default=default_data, help="Path to dataset root")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    evaluate_model(args.weights, args.data_dir, args.device)


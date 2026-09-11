"""
EfficientNet-B0 Transfer Learning Training Pipeline for Mango Leaf Disease Classification.
Executes fine-tuning with Cross-Entropy loss, AdamW optimizer, Cosine Annealing, and checkpointing.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys

# Ensure backend directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "training"))

from dataset import create_dataloaders, MANGO_CLASSES, CLASS_TO_IDX
from classifier import build_efficientnet_classifier


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
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

    epoch_loss = running_loss / max(1, total)
    epoch_acc = (correct / max(1, total)) * 100.0
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / max(1, total)
    val_acc = (correct / max(1, total)) * 100.0
    return val_loss, val_acc


def main():
    default_data_dir = os.path.join(BASE_DIR, "data", "Mango S data")
    default_save_path = os.path.join(BASE_DIR, "models", "mango_cnn_efficientnet.pth")

    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on Mango Leaf Disease Dataset")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Path to dataset root folder")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--max_samples", type=int, default=200, help="Max images per class (default 200)")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="L2 weight decay regularization")
    parser.add_argument("--save_path", type=str, default=default_save_path, help="Output weights file path")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 60, flush=True)
    print(f"EfficientNet-B0 Fast Mango Leaf Disease Training Engine", flush=True)
    print(f"Device: {device} | Classes: {len(MANGO_CLASSES)}", flush=True)
    print(f"Data directory: {args.data_dir}", flush=True)
    print(f"Max samples per class: {args.max_samples}", flush=True)
    print(f"Save weights to: {args.save_path}", flush=True)
    print("=" * 60, flush=True)

    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset directory '{args.data_dir}' not found.", flush=True)
        return

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_ratio=0.15,
        test_ratio=0.15,
        max_samples_per_class=args.max_samples,
        seed=42
    )

    if len(train_loader.dataset) == 0:
        print("Error: No training samples found. Please check dataset folder structure.", flush=True)
        return

    print(f"Dataset split -> Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}", flush=True)
    print(f"Class mapping: {CLASS_TO_IDX}\n", flush=True)

    # Build model with pretrained backbone
    model = build_efficientnet_classifier(num_classes=len(MANGO_CLASSES), pretrained=True)
    model = model.to(device)

    # Differential learning rate: lower lr for feature backbone, higher for classifier head
    classifier_params = list(model.classifier.parameters())
    feature_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]

    optimizer = optim.AdamW([
        {"params": feature_params, "lr": args.lr * 0.2},
        {"params": classifier_params, "lr": args.lr}
    ], weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start_t

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%", flush=True)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_name": "EfficientNet-B0",
                "state_dict": model.state_dict(),
                "classes": MANGO_CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "best_val_acc": best_val_acc
            }
            torch.save(checkpoint, args.save_path)
            print(f"  --> Saved new best checkpoint to '{args.save_path}' (Val Acc: {best_val_acc:.2f}%)", flush=True)

    # Final evaluation on hold-out test set
    if len(test_loader.dataset) > 0 and os.path.exists(args.save_path):
        print("\n--- Evaluating Best Model on Unseen Hold-out Test Set ---", flush=True)
        best_ckpt = torch.load(args.save_path, map_location=device)
        model.load_state_dict(best_ckpt["state_dict"])
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Final Test Accuracy: {test_acc:.2f}% (Loss: {test_loss:.4f})", flush=True)

    print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc:.2f}%. Model saved to '{args.save_path}'.", flush=True)


if __name__ == "__main__":
    main()



import os
import sys
import shutil
import argparse
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ROBOFLOW_YAML = os.path.join(PROJECT_ROOT, "backend", "data", "Multiclass_leaf.v5i.yolov8-data", "data.yaml")
OUTPUT_MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "models", "mango_yolo.pt")

def train_and_export_yolo(
    epochs: int = 10,
    imgsz: int = 320,
    batch: int = 16,
    model_type: str = "yolov8n-obb.pt",
    lr0: float = 0.002,
    resume: bool = False
):
    """
    Trains YOLOv8-OBB on the Roboflow ground-truth Mango Leaf dataset,
    evaluates on validation & test splits, and exports best weights to backend/models/mango_yolo.pt.
    """
    from ultralytics import YOLO

    if not os.path.exists(ROBOFLOW_YAML):
        raise FileNotFoundError(f"Roboflow data.yaml not found at: {ROBOFLOW_YAML}")

    print("=" * 70)
    print("🚀 MANGO GUARD AI - YOLOv8 OBB TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Dataset YAML  : {ROBOFLOW_YAML}")
    print(f"  Base Model    : {model_type}")
    print(f"  Total Epochs  : {epochs}")
    print(f"  Image Size    : {imgsz}x{imgsz}")
    print(f"  Batch Size    : {batch}")
    print(f"  Initial LR    : {lr0}")
    print(f"  Output Export : {OUTPUT_MODEL_PATH}")
    print("=" * 70)

    model = YOLO(model_type)

    results = model.train(
        data=ROBOFLOW_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=0,
        optimizer="AdamW",
        lr0=lr0,
        name="mango_yolo_roboflow",
        save=True,
        device="cpu",
        exist_ok=True,
        resume=resume
    )

    print("\n" + "=" * 70)
    print("📊 EVALUATING BEST MODEL ON VALIDATION SPLIT")
    print("=" * 70)
    val_metrics = model.val(data=ROBOFLOW_YAML, imgsz=imgsz, batch=batch, split="val")

    print("\n" + "=" * 70)
    print("🧪 EVALUATING BEST MODEL ON TEST SPLIT")
    print("=" * 70)
    test_metrics = model.val(data=ROBOFLOW_YAML, imgsz=imgsz, batch=batch, split="test")

    # Export trained weights
    save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir:
        best_weights_path = os.path.join(save_dir, "weights", "best.pt")
        if os.path.exists(best_weights_path):
            os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
            shutil.copy2(best_weights_path, OUTPUT_MODEL_PATH)
            print(f"\n✅ Successfully exported best trained weights -> '{OUTPUT_MODEL_PATH}'")
        else:
            print(f"⚠️ best.pt not found in {save_dir}")

    print("\n" + "=" * 70)
    print("🎉 YOLOv8 TRAINING & EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Mango Leaf Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--weights", type=str, default="yolov8n-obb.pt", help="Base weights path")
    parser.add_argument("--lr", type=float, default=0.002, help="Initial learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    args = parser.parse_args()

    train_and_export_yolo(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_type=args.weights,
        lr0=args.lr,
        resume=args.resume
    )


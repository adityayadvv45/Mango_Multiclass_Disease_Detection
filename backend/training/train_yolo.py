"""
🎯 MANGO GUARD AI - YOLOv8 TRAINING & EVALUATION SCRIPT (Ultralytics)
Trains YOLOv8 for foliar pathogen lesion detection & multi-disease localization,
evaluates mAP/precision/recall, exports ONNX, and logs metrics.
"""

import os
import sys
import yaml
import csv
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def generate_yolo_dataset_yaml(output_path="mango_lesion_data.yaml"):
    """
    Creates YOLOv8 dataset configuration YAML file.
    """
    dataset_dir = os.path.join(PROJECT_ROOT, "backend", "data", "Mango_YOLO_Lesions")
    config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'Anthracnose',
            1: 'Bacterial Canker',
            2: 'Cutting Weevil',
            3: 'Die Back',
            4: 'Gall Midge',
            5: 'Healthy Foliage',
            6: 'Powdery Mildew',
            7: 'Sooty Mold'
        }
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ Generated YOLOv8 dataset configuration: {output_path}")
    return output_path

def train_yolo(epochs=50, imgsz=640, batch=16, model_type="yolov8n.pt"):
    """
    Executes YOLOv8 training on Mango Leaf Disease dataset if ultralytics is available.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️ Ultralytics is not installed in the current environment.")
        print("   To install, run: pip install ultralytics")
        return None
        
    yaml_path = generate_yolo_dataset_yaml()
    print("=" * 60)
    print(f"🚀 Initializing YOLOv8 Model ({model_type}) for {epochs} Epochs...")
    print("=" * 60)
    
    model = YOLO(model_type)
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        augment=True,
        name="mango_yolo_run",
        save=True
    )
    
    print("\n--- Validating YOLOv8 Model Performance ---")
    metrics = model.val()
    print(f"mAP@50     : {metrics.box.map50 * 100:.2f}%")
    print(f"mAP@50-95  : {metrics.box.map * 100:.2f}%")
    print(f"Precision  : {metrics.box.mp * 100:.2f}%")
    print(f"Recall     : {metrics.box.mr * 100:.2f}%")
    
    # Export ONNX model
    onnx_path = model.export(format="onnx")
    print(f"✅ Exported YOLOv8 ONNX model: {onnx_path}")
    return model

if __name__ == "__main__":
    generate_yolo_dataset_yaml()

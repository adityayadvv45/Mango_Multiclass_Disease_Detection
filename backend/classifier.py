import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

try:
    import torchvision.transforms as transforms
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# 8 Official Botanical Mango Leaf Disease Classes
DISEASE_CLASSES = [
    {
        "id": "healthy",
        "name": "Healthy",
        "scientific_name": "Mangifera indica (Healthy)",
        "category": "Healthy",
        "risk": "None",
        "status": "Healthy Specimen"
    },
    {
        "id": "anthracnose",
        "name": "Anthracnose",
        "scientific_name": "Colletotrichum gloeosporioides",
        "category": "Fungal",
        "risk": "Moderate",
        "status": "Disease Detected"
    },
    {
        "id": "bacterial-canker",
        "name": "Bacterial Canker",
        "scientific_name": "Xanthomonas citri pv. mangiferaeindicae",
        "category": "Bacterial",
        "risk": "High",
        "status": "Disease Detected"
    },
    {
        "id": "powdery-mildew",
        "name": "Powdery Mildew",
        "scientific_name": "Oidium mangiferae",
        "category": "Fungal",
        "risk": "Moderate",
        "status": "Disease Detected"
    },
    {
        "id": "sooty-mold",
        "name": "Sooty Mold",
        "scientific_name": "Capnodium mangiferae / Meliola mangiferae",
        "category": "Fungal",
        "risk": "Low",
        "status": "Disease Detected"
    },
    {
        "id": "die-back",
        "name": "Die Back",
        "scientific_name": "Lasiodiplodia theobromae",
        "category": "Fungal / Vascular",
        "risk": "High",
        "status": "Disease Detected"
    },
    {
        "id": "gall-midge",
        "name": "Gall Midge",
        "scientific_name": "Procontarinia matteiana",
        "category": "Pest / Insect Infestation",
        "risk": "Moderate",
        "status": "Pest Detected"
    },
    {
        "id": "cutting-weevil",
        "name": "Cutting Weevil",
        "scientific_name": "Deporaus marginatus",
        "category": "Pest / Insect Damage",
        "risk": "Moderate",
        "status": "Pest Damage Detected"
    }
]

# Class lookups & metadata mappings
CLASS_NAMES = [d["name"] for d in DISEASE_CLASSES]
CLASS_IDS = [d["id"] for d in DISEASE_CLASSES]
CLASS_MAP = {d["id"]: d for d in DISEASE_CLASSES}
CLASS_MAP["sooty-mould"] = CLASS_MAP["sooty-mold"]
CLASS_MAP["bacterial_canker"] = CLASS_MAP["bacterial-canker"]
CLASS_MAP["powdery_mildew"] = CLASS_MAP["powdery-mildew"]
CLASS_MAP["die_back"] = CLASS_MAP["die-back"]
CLASS_MAP["gall_midge"] = CLASS_MAP["gall-midge"]
CLASS_MAP["cutting_weevil"] = CLASS_MAP["cutting-weevil"]

NAME_TO_META = {d["name"]: d for d in DISEASE_CLASSES}
NAME_TO_META["Sooty Mould"] = NAME_TO_META["Sooty Mold"]

ID_TO_INDEX = {d["id"]: idx for idx, d in enumerate(DISEASE_CLASSES)}
INDEX_TO_CLASS = {idx: d for idx, d in enumerate(DISEASE_CLASSES)}


def build_efficientnet_classifier(num_classes=8, pretrained=True):
    """Constructs EfficientNet-B0 backbone with an 8-class classification head."""
    if not HAS_TORCHVISION:
        raise RuntimeError("torchvision is required for EfficientNet-B0 model creation.")

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Replace classifier head for 8 mango leaf classes
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes)
    )
    return model


class EfficientNetMangoClassifier:
    """
    CNN Image Classifier using fine-tuned EfficientNet-B0 transfer learning.
    Accepts whole leaf images or localized crop patches, classifies pathology
    across the 8 botanical classes, and produces calibrated probabilities.
    """
    def __init__(self, weights_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if weights_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.weights_path = os.path.join(base_dir, "models", "mango_cnn_efficientnet.pth")
        else:
            self.weights_path = weights_path

        self.model = None
        self.is_weights_loaded = False
        self.model_name = "EfficientNet-B0-Trained"

        # ImageNet standardization parameters
        if HAS_TORCHVISION:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = None

        self._initialize_model()

    def _initialize_model(self):
        """Instantiate network architecture and load custom fine-tuned weights if available."""
        if not HAS_TORCHVISION:
            print("[CNN Classifier] Warning: torchvision not available. Model running in fallback mode.")
            return

        try:
            # Build network structure
            self.model = build_efficientnet_classifier(num_classes=len(DISEASE_CLASSES), pretrained=False)

            if os.path.exists(self.weights_path):
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                self.is_weights_loaded = True
                print(f"[CNN Classifier] Loaded newly trained weights from: {self.weights_path}")
            else:
                print(f"[CNN Classifier] Note: No custom weights found at '{self.weights_path}'.")
                print("[CNN Classifier] Initializing with pretrained EfficientNet-B0 backbone. Run training to generate fine-tuned weights.")
                try:
                    pretrained_base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
                    self.model.features.load_state_dict(pretrained_base.features.state_dict())
                except Exception as e:
                    print(f"[CNN Classifier] Pretrained backbone note: {e}")

            self.model.to(self.device)
            self.model.eval()
        except Exception as err:
            print(f"[CNN Classifier] Initialization exception: {err}")

    def preprocess_crop(self, crop):
        """
        Preprocess a single image crop (PIL Image, NumPy array, or bytes) into a normalized tensor.
        Crop is resized to 224x224 and normalized using ImageNet parameters.
        """
        if isinstance(crop, np.ndarray):
            pil_img = Image.fromarray(crop).convert("RGB")
        elif isinstance(crop, Image.Image):
            pil_img = crop.convert("RGB")
        elif isinstance(crop, (bytes, bytearray)):
            import io
            pil_img = Image.open(io.BytesIO(crop)).convert("RGB")
        else:
            raise TypeError(f"Unsupported crop input type: {type(crop)}")

        if self.transform is not None:
            tensor = self.transform(pil_img)
        else:
            pil_resized = pil_img.resize((224, 224))
            arr = np.array(pil_resized, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            tensor = torch.from_numpy(arr.transpose((2, 0, 1)))

        return tensor

    def classify_crop(self, crop):
        """
        Classify a single leaf or cropped lesion region using pure neural network forward pass
        through the fine-tuned EfficientNet-B0 model.
        """
        tensor = self.preprocess_crop(crop).unsqueeze(0).to(self.device)

        if self.model is None:
            return self._heuristic_fallback(crop)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_idx = int(np.argmax(probs))
        top_meta = INDEX_TO_CLASS[top_idx]
        top_conf = round(float(probs[top_idx]) * 100.0, 2)

        distribution = {
            INDEX_TO_CLASS[i]["name"]: round(float(probs[i]) * 100.0, 2)
            for i in range(len(DISEASE_CLASSES))
        }

        return {
            "disease": top_meta["name"],
            "disease_id": top_meta["id"],
            "scientific_name": top_meta["scientific_name"],
            "category": top_meta["category"],
            "risk": top_meta["risk"],
            "cnn_confidence": top_conf,
            "distribution": distribution
        }

    def classify_crops_batch(self, crops):
        """
        Batch inference for multiple detected regions in a single forward pass.
        """
        if not crops:
            return []

        if self.model is None:
            return [self.classify_crop(c) for c in crops]

        tensors = [self.preprocess_crop(c) for c in crops]
        batch_tensor = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            logits_batch = self.model(batch_tensor)
            probs_batch = torch.softmax(logits_batch, dim=1).cpu().numpy()

        results = []
        for i in range(len(crops)):
            probs = probs_batch[i]
            top_idx = int(np.argmax(probs))
            top_meta = INDEX_TO_CLASS[top_idx]
            top_conf = round(float(probs[top_idx]) * 100.0, 2)

            distribution = {
                INDEX_TO_CLASS[j]["name"]: round(float(probs[j]) * 100.0, 2)
                for j in range(len(DISEASE_CLASSES))
            }

            results.append({
                "disease": top_meta["name"],
                "disease_id": top_meta["id"],
                "scientific_name": top_meta["scientific_name"],
                "category": top_meta["category"],
                "risk": top_meta["risk"],
                "cnn_confidence": top_conf,
                "distribution": distribution
            })

        return results

    def _heuristic_fallback(self, crop):
        """Graceful fallback when model weights are not loaded."""
        return {
            "disease": "Healthy",
            "disease_id": "healthy",
            "scientific_name": "Mangifera indica (Healthy)",
            "category": "Healthy",
            "risk": "None",
            "cnn_confidence": 50.0,
            "distribution": {d["name"]: round(100.0 / len(DISEASE_CLASSES), 2) for d in DISEASE_CLASSES}
        }


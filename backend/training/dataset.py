import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision.transforms as transforms
    HAS_TRANSFORMS = True
except ImportError:
    HAS_TRANSFORMS = False


# 8 Canonical Botanical Mango Leaf Disease Classes
MANGO_CLASSES = [
    "Healthy",
    "Anthracnose",
    "Bacterial Canker",
    "Powdery Mildew",
    "Sooty Mold",
    "Die Back",
    "Gall Midge",
    "Cutting Weevil"
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(MANGO_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(MANGO_CLASSES)}

# Normalize folder name variations and aliases to canonical class names
FOLDER_ALIAS_MAP = {
    "healthy": "Healthy",
    "anthracnose": "Anthracnose",
    "bacterial canker": "Bacterial Canker",
    "bacterial_canker": "Bacterial Canker",
    "bacterial-canker": "Bacterial Canker",
    "powdery mildew": "Powdery Mildew",
    "powdery_mildew": "Powdery Mildew",
    "powdery-mildew": "Powdery Mildew",
    "sooty mold": "Sooty Mold",
    "sooty_mold": "Sooty Mold",
    "sooty-mold": "Sooty Mold",
    "sooty mould": "Sooty Mold",
    "sooty_mould": "Sooty Mold",
    "sooty-mould": "Sooty Mold",
    "die back": "Die Back",
    "die_back": "Die Back",
    "die-back": "Die Back",
    "gall midge": "Gall Midge",
    "gall_midge": "Gall Midge",
    "gall-midge": "Gall Midge",
    "cutting weevil": "Cutting Weevil",
    "cutting_weevil": "Cutting Weevil",
    "cutting-weevil": "Cutting Weevil"
}


def get_training_transforms(image_size=224):
    """Rich data augmentation pipeline for training EfficientNet-B0 on mango leaf images."""
    if not HAS_TRANSFORMS:
        raise RuntimeError("torchvision.transforms is required for data augmentation.")

    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_validation_transforms(image_size=224):
    """Standard evaluation transformation for validation and testing."""
    if not HAS_TRANSFORMS:
        raise RuntimeError("torchvision.transforms is required for validation transforms.")

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class MangoLeafDataset(Dataset):
    """
    PyTorch Dataset for mango leaf pathology images.
    Accepts either an explicit sample list [(filepath, label_idx), ...]
    or automatically scans class subfolders within root_dir.
    """
    def __init__(self, samples=None, root_dir=None, transform=None):
        self.transform = transform
        self.samples = []

        if samples is not None:
            self.samples = samples
        elif root_dir and os.path.exists(root_dir):
            valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            for entry in os.listdir(root_dir):
                entry_path = os.path.join(root_dir, entry)
                if os.path.isdir(entry_path):
                    norm_key = entry.lower().strip()
                    canonical_name = FOLDER_ALIAS_MAP.get(norm_key, None)
                    if canonical_name and canonical_name in CLASS_TO_IDX:
                        class_idx = CLASS_TO_IDX[canonical_name]
                        for fname in os.listdir(entry_path):
                            ext = os.path.splitext(fname)[1].lower()
                            if ext in valid_extensions:
                                self.samples.append((os.path.join(entry_path, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(34, 197, 94))

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def scan_all_samples(data_dir):
    """Discovers all images from class folders in data_dir and maps them to canonical labels."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    class_samples = {cls_name: [] for cls_name in MANGO_CLASSES}

    if not os.path.exists(data_dir):
        return class_samples

    for entry in os.listdir(data_dir):
        entry_path = os.path.join(data_dir, entry)
        if os.path.isdir(entry_path):
            norm_key = entry.lower().strip()
            canonical_name = FOLDER_ALIAS_MAP.get(norm_key, None)
            if canonical_name and canonical_name in class_samples:
                for fname in os.listdir(entry_path):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in valid_extensions:
                        class_samples[canonical_name].append(os.path.join(entry_path, fname))

    return class_samples


def create_dataloaders(data_dir, batch_size=32, val_ratio=0.15, test_ratio=0.15, max_samples_per_class=200, seed=42, num_workers=0):
    """
    Constructs train, validation, and test PyTorch DataLoaders with stratified splitting.
    Caps each class to max_samples_per_class (default 200) for fast, robust training.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_dataset = MangoLeafDataset(root_dir=train_dir, transform=get_training_transforms())
        val_dataset = MangoLeafDataset(root_dir=val_dir, transform=get_validation_transforms())
        test_dir = os.path.join(data_dir, "test")
        test_dataset = MangoLeafDataset(root_dir=test_dir, transform=get_validation_transforms()) if os.path.exists(test_dir) else val_dataset

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    # Stratified split from class folders in data_dir
    class_samples = scan_all_samples(data_dir)
    random.seed(seed)

    train_samples = []
    val_samples = []
    test_samples = []

    for cls_name, paths in class_samples.items():
        shuffled = list(paths)
        random.shuffle(shuffled)
        if max_samples_per_class and len(shuffled) > max_samples_per_class:
            shuffled = shuffled[:max_samples_per_class]

        n = len(shuffled)
        if n == 0:
            continue

        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        n_train = n - n_val - n_test

        cls_idx = CLASS_TO_IDX[cls_name]
        train_samples.extend([(p, cls_idx) for p in shuffled[:n_train]])
        val_samples.extend([(p, cls_idx) for p in shuffled[n_train:n_train + n_val]])
        test_samples.extend([(p, cls_idx) for p in shuffled[n_train + n_val:]])

    train_dataset = MangoLeafDataset(samples=train_samples, transform=get_training_transforms())
    val_dataset = MangoLeafDataset(samples=val_samples, transform=get_validation_transforms())
    test_dataset = MangoLeafDataset(samples=test_samples, transform=get_validation_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader



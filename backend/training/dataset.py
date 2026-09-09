import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision.transforms as transforms
    HAS_TRANSFORMS = True
except ImportError:
    HAS_TRANSFORMS = False


# 8 Canonical Mango Leaf Classes
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


def get_training_transforms(image_size=224):
    """Data augmentation pipeline for training EfficientNet-B0 on leaf crops."""
    if not HAS_TRANSFORMS:
        raise RuntimeError("torchvision.transforms is required for data augmentation.")

    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_validation_transforms(image_size=224):
    """Standard evaluation transformation for validation/testing."""
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


class MangoLeafCropDataset(Dataset):
    """
    PyTorch Dataset for cropped mango leaf pathology patches.
    Directory structure expected:
      root_dir/
        ├── Healthy/
        ├── Anthracnose/
        ├── Bacterial Canker/
        ├── Powdery Mildew/
        ├── Sooty Mold/
        ├── Die Back/
        ├── Gall Midge/
        └── Cutting Weevil/
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        if os.path.exists(root_dir):
            for class_name, class_idx in CLASS_TO_IDX.items():
                class_dir = os.path.join(root_dir, class_name)
                # Also try lowercase / hyphenated folder names
                if not os.path.exists(class_dir):
                    alt_name = class_name.lower().replace(" ", "-")
                    class_dir = os.path.join(root_dir, alt_name)

                if os.path.isdir(class_dir):
                    for fname in os.listdir(class_dir):
                        ext = os.path.splitext(fname)[1].lower()
                        if ext in valid_extensions:
                            self.samples.append((os.path.join(class_dir, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def create_dataloaders(data_dir, batch_size=32, num_workers=0):
    """Constructs train, validation, and test PyTorch DataLoaders."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = MangoLeafCropDataset(train_dir, transform=get_training_transforms())
    val_dataset = MangoLeafCropDataset(val_dir, transform=get_validation_transforms())
    test_dataset = MangoLeafCropDataset(test_dir, transform=get_validation_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

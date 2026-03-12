import os
import random
from typing import Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def clean_class_name(dirname: str) -> str:
    if "-" in dirname:
        return dirname.split("-")[1]
    return dirname


class DogDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, class_id


def get_loaders(
        root: str,
        batch_size: int = 32,
        img_size: int = 224,
        num_workers: int = 0   # ← фиксировано для Windows
):

    if not os.path.exists(root):
        raise FileNotFoundError(f"Путь не найден: {root}")

    class_dirs = [d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d))]

    if len(class_dirs) == 0:
        raise RuntimeError("Не найдено ни одного класса в директориях.")

    class_names = [clean_class_name(d) for d in class_dirs]

    class_to_idx = {cls_dir: idx for idx, cls_dir in enumerate(class_dirs)}

    samples = []
    extensions = (".jpg", ".jpeg", ".png", ".bmp")

    for cls_dir in class_dirs:
        cls_path = os.path.join(root, cls_dir)
        class_id = class_to_idx[cls_dir]

        for fname in os.listdir(cls_path):
            if fname.lower().endswith(extensions):
                samples.append((os.path.join(cls_path, fname), class_id))

    if len(samples) == 0:
        raise RuntimeError("В папках классов не найдено изображений.")

    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = DogDataset(train_samples, transform=train_transform)
    val_dataset = DogDataset(val_samples, transform=test_val_transform)
    test_dataset = DogDataset(test_samples, transform=test_val_transform)

    # --- СТАБИЛЬНЫЕ DataLoader-ы ДЛЯ WINDOWS ---
    loader_args = dict(
        batch_size=batch_size,
        num_workers=0,      # ← критично для Windows + Python 3.12
        pin_memory=True
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_args
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_args
    )

    return train_loader, val_loader, test_loader, class_names

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import (
    CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS,
    SEED, SPLIT_FILE,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_class_directory(search_root: str | Path) -> Path:
    """Find the directory whose immediate subdirectories are the nine labels."""
    root = Path(search_root)
    expected = {name.casefold() for name in CLASS_NAMES}
    candidates = [root] if root.is_dir() else []
    candidates.extend(path for path in root.rglob("*") if path.is_dir())
    for candidate in candidates:
        try:
            children = {p.name.casefold() for p in candidate.iterdir() if p.is_dir()}
        except PermissionError:
            continue
        if expected.issubset(children):
            return candidate
    raise FileNotFoundError(
        f"Could not find a folder containing all nine class folders under: {root}\n"
        f"Expected: {', '.join(CLASS_NAMES)}"
    )


def discover_images(class_root: str | Path) -> pd.DataFrame:
    class_root = Path(class_root)
    records: list[dict[str, str]] = []
    name_map = {name.casefold(): name for name in CLASS_NAMES}
    for class_dir in class_root.iterdir():
        if not class_dir.is_dir() or class_dir.name.casefold() not in name_map:
            continue
        label = name_map[class_dir.name.casefold()]
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    records.append({
                        "image_path": image_path.relative_to(class_root).as_posix(),
                        "label": label,
                    })
                except (UnidentifiedImageError, OSError):
                    print(f"Skipping unreadable image: {image_path}")
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(f"No readable images found in {class_root}")
    return frame.sort_values(["label", "image_path"]).reset_index(drop=True)


def create_splits(class_root: str | Path, split_file: str | Path = SPLIT_FILE,
                  force: bool = False, seed: int = SEED) -> pd.DataFrame:
    split_file = Path(split_file)
    if split_file.exists() and not force:
        frame = pd.read_csv(split_file)
        missing = [
            str(Path(class_root) / relative)
            for relative in frame["image_path"].tolist()
            if not (Path(class_root) / relative).is_file()
        ]
        if not missing:
            return frame
        print("Existing split file does not match this dataset; rebuilding it.")

    frame = discover_images(class_root)
    class_counts = frame["label"].value_counts()
    missing_classes = [name for name in CLASS_NAMES if name not in class_counts.index]
    if missing_classes or (not class_counts.empty and int(class_counts.min()) < 3):
        raise ValueError(
            "A stratified train/validation/test split requires at least three valid "
            "images in every class. Missing or under-represented classes: "
            f"{missing_classes or class_counts[class_counts < 3].to_dict()}"
        )
    test_count = int(np.ceil(len(frame) * 0.15))
    validation_count = int(np.ceil((len(frame) - test_count) * (0.15 / 0.85)))
    if test_count < len(CLASS_NAMES) or validation_count < len(CLASS_NAMES):
        raise ValueError(
            "The dataset is too small to place every class in both validation and "
            f"test partitions. Found {len(frame)} images; use the full RealWaste dataset."
        )
    train_frame, test_frame = train_test_split(
        frame, test_size=0.15, random_state=seed, stratify=frame["label"]
    )
    # 0.17647 of the remaining 85% gives approximately 15% of the full dataset.
    train_frame, val_frame = train_test_split(
        train_frame, test_size=(0.15 / 0.85), random_state=seed,
        stratify=train_frame["label"]
    )
    train_frame = train_frame.assign(split="train")
    val_frame = val_frame.assign(split="validation")
    test_frame = test_frame.assign(split="test")
    result = pd.concat([train_frame, val_frame, test_frame], ignore_index=True)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(split_file, index=False)
    print("Saved stratified split:")
    print(result.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    return result


class WasteImageDataset(Dataset):
    def __init__(self, class_root: str | Path, frame: pd.DataFrame,
                 transform=None):
        self.class_root = Path(class_root)
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
        self.targets = [self.class_to_idx[label] for label in self.frame["label"]]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.class_root / row["image_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[row["label"]]


def get_transforms(image_size: int = IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.18, contrast=0.18,
                               saturation=0.15, hue=0.025),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def make_loaders(class_root: str | Path, split_file: str | Path = SPLIT_FILE,
                 image_size: int = IMAGE_SIZE, batch_size: int = 24,
                 num_workers: int = NUM_WORKERS, seed: int = SEED):
    split_frame = pd.read_csv(split_file)
    train_transform, eval_transform = get_transforms(image_size)
    loaders = {}
    for split in ("train", "validation", "test"):
        subset = split_frame[split_frame["split"] == split].copy()
        dataset = WasteImageDataset(
            class_root, subset,
            transform=train_transform if split == "train" else eval_transform,
        )
        if split == "train":
            counts = np.bincount(dataset.targets, minlength=len(CLASS_NAMES))
            weights = [1.0 / max(counts[target], 1) for target in dataset.targets]
            generator = torch.Generator().manual_seed(seed)
            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True,
                generator=generator
            )
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )
        else:
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )
    return loaders

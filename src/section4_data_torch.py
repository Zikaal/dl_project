from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


@dataclass
class DataCfg:
    root: str = "./data"
    img_size: int = 160
    batch_size: int = 32
    num_workers: int = 0  # Windows friendly
    val_ratio: float = 0.2
    seed: int = 42


def _make_transforms(img_size: int, augment: bool):
    # ImageNet normalization (важно для pretrained ResNet)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),   # 1
            transforms.RandomHorizontalFlip(p=0.5),                      # 2
            transforms.RandomRotation(10),                               # 3
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),  # 4
            transforms.RandomGrayscale(p=0.1),                           # 5
            transforms.ToTensor(),
            normalize,
])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, test_tf


def make_loaders(cfg: DataCfg, augment: bool, max_train: int = 2000, max_val: int = 500):
    train_tf, test_tf = _make_transforms(cfg.img_size, augment=augment)

    full_train = OxfordIIITPet(
        root=cfg.root, split="trainval",
        target_types="binary-category",
        download=True, transform=train_tf
    )
    full_val = OxfordIIITPet(
        root=cfg.root, split="trainval",
        target_types="binary-category",
        download=True, transform=test_tf
    )

    n = len(full_train)
    n_val = int(n * cfg.val_ratio)
    n_train = n - n_val

    g = torch.Generator().manual_seed(cfg.seed)
    train_idx, val_idx = random_split(range(n), [n_train, n_val], generator=g)

    train_idx = list(train_idx)[:max_train]
    val_idx = list(val_idx)[:max_val]

    train_ds = torch.utils.data.Subset(full_train, train_idx)
    val_ds = torch.utils.data.Subset(full_val, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=False)

    return train_loader, val_loader

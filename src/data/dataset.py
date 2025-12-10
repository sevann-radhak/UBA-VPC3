import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple


class CustomDataset(Dataset):
    def __init__(
        self,
        images: list,
        labels: list,
        transform: Optional[transforms.Compose] = None
    ):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            image = self.images[idx]
            label = self.labels[idx]

            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx}: {str(e)}")


def get_optimal_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    if os.name == 'nt':
        return min(2, cpu_count)
    else:
        return min(4, cpu_count - 1)


def get_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if num_workers is None:
        num_workers = get_optimal_num_workers()
    
    pin_memory = device.startswith('cuda') and torch.cuda.is_available()
    persistent_workers = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False
    )

    return train_loader, val_loader, test_loader





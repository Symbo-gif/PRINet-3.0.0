"""Dataset utilities for standard image-classification benchmarks.

Provides convenient loaders for CIFAR-10 and Fashion-MNIST that
integrate with PRINet training pipelines.  All data is cached in
``~/.cache/prinet/datasets`` by default.

Example::

    from prinet.utils.datasets import get_cifar10_loaders

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    for x, y in train_loader:
        # x: (B, 3, 32, 32), y: (B,)
        ...

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_DATA_ROOT = Path.home() / ".cache" / "prinet" / "datasets"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

FMNIST_MEAN = (0.2860,)
FMNIST_STD = (0.3530,)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

FMNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _data_root(root: Optional[str | Path]) -> Path:
    if root is None:
        root = _DEFAULT_DATA_ROOT
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: Optional[str | Path] = None,
    augment_train: bool = True,
    download: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    """Return (train_loader, test_loader) for CIFAR-10.

    Args:
        batch_size: Batch size for both loaders.
        num_workers: Worker processes for DataLoader.
        data_root: Directory in which to cache the dataset.
        augment_train: Whether to apply random crop + horizontal flip.
        download: Automatically download if not present.
        pin_memory: Pin memory tensors (faster GPU transfer).

    Returns:
        Tuple ``(train_loader, test_loader)``.
    """
    import torchvision
    import torchvision.transforms as T

    root = _data_root(data_root)

    train_tf_list = []
    if augment_train:
        train_tf_list += [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
        ]
    train_tf_list += [T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    train_tf = T.Compose(train_tf_list)

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=str(root), train=True, transform=train_tf, download=download
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=str(root), train=False, transform=test_tf, download=download
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Fashion-MNIST
# ---------------------------------------------------------------------------


def get_fashion_mnist_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: Optional[str | Path] = None,
    augment_train: bool = True,
    download: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    """Return (train_loader, test_loader) for Fashion-MNIST.

    Fashion-MNIST images are 1-channel 28×28.  The loader resizes them
    to 32×32 to match the CIFAR-10 spatial resolution expected by
    :class:`~prinet.nn.hybrid.HybridPRINetV2` with ``use_conv_stem=True``.

    Args:
        batch_size: Batch size for both loaders.
        num_workers: Worker processes for DataLoader.
        data_root: Directory in which to cache the dataset.
        augment_train: Whether to apply random horizontal flip.
        download: Automatically download if not present.
        pin_memory: Pin memory tensors.

    Returns:
        Tuple ``(train_loader, test_loader)``.
    """
    import torchvision
    import torchvision.transforms as T

    root = _data_root(data_root)

    train_tf_list: list[Any] = [T.Resize(32), T.Grayscale(num_output_channels=3)]
    if augment_train:
        train_tf_list.append(T.RandomHorizontalFlip())
    train_tf_list += [T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    train_tf = T.Compose(train_tf_list)

    test_tf = T.Compose([
        T.Resize(32),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = torchvision.datasets.FashionMNIST(
        root=str(root), train=True, transform=train_tf, download=download
    )
    test_ds = torchvision.datasets.FashionMNIST(
        root=str(root), train=False, transform=test_tf, download=download
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Quick evaluation helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: Optional[torch.device] = None,
) -> float:
    """Compute top-1 accuracy of *model* on *loader*.

    Args:
        model: Model whose ``forward`` returns log-probabilities or logits.
        loader: Data loader over ``(inputs, labels)`` batches.
        device: Target device; auto-detected if ``None``.

    Returns:
        Float accuracy in ``[0, 1]``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(inputs)
        preds = out.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)

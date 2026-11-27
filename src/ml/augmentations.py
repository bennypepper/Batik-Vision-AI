"""Domain-safe uint8 Batik augmentation pipelines for v3/v4 training.

Key guarantees:
1. Images are loaded at native resolution as RGB ``uint8`` and passed into
   Albumentations as ``uint8`` (never pre-resized to 224x224 before
   ``RandomResizedCrop`` and never passed as ``float32 [0, 255]``).
2. Cast to ``float32`` in raw ``[0, 255]`` range happens strictly *after*
   Albumentations, preserving exact parity with Keras ``Rescaling(1/127.5, -1.0)``.
3. Domain-destructive transforms (``GridDistortion``, ``ElasticTransform``) are
   excluded and HSV shifts are bounded (hue <= 5, sat <= 15, val <= 15) to
   preserve Sogan brown and Lasem red motif color signatures.
4. Orientation-sensitive motifs (Parang diagonal polarity, Lontara script)
   disable ``HorizontalFlip``.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np

ORIENTATION_SENSITIVE_CLASSES: frozenset[str] = frozenset({
    "Solo_Parang",
    "Yogyakarta_Parang",
    "Sulawesi_Selatan_Lontara",
})


def is_orientation_sensitive(class_name: str) -> bool:
    """Return True if horizontal flipping corrupts the motif's semantic identity."""
    return class_name in ORIENTATION_SENSITIVE_CLASSES


def load_image_uint8(filepath: str, fallback_size: int = 224) -> np.ndarray:
    """Load an image from disk at native resolution in RGB ``uint8`` format.

    If the file cannot be decoded, returns a black ``uint8`` placeholder of
    shape ``(fallback_size, fallback_size, 3)``.
    """
    img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return np.zeros((fallback_size, fallback_size, 3), dtype=np.uint8)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def build_train_augmentation(
    img_size: int = 224,
    tier: str = "tier2",
    allow_horizontal_flip: bool = True,
) -> A.Compose:
    """Build a Batik domain-safe Albumentations training pipeline operating on ``uint8``."""
    transforms: list[A.BasicTransform] = []
    if allow_horizontal_flip:
        transforms.append(A.HorizontalFlip(p=0.5))

    if tier == "tier1":
        transforms.extend([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.5),
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.75, 1.0), ratio=(0.9, 1.1), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                p=0.25,
            ),
            A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.25),
            A.Sharpen(alpha=(0.1, 0.25), lightness=(0.85, 1.15), p=0.2),
        ])
    elif tier == "tier2":
        transforms.extend([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.4),
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.4),
            A.GaussianBlur(blur_limit=(3, 3), p=0.15),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        ])
    elif tier == "tier3":
        transforms.extend([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=12, val_shift_limit=10, p=0.3),
        ])
    else:
        raise ValueError(f"Unsupported augmentation tier: {tier!r}")

    transforms.append(A.Resize(height=img_size, width=img_size))
    return A.Compose(transforms)


def build_eval_augmentation(img_size: int = 224) -> A.Compose:
    """Build deterministic resize-only evaluation pipeline operating on ``uint8``."""
    return A.Compose([A.Resize(height=img_size, width=img_size)])


def apply_augmentation_to_float32(img_uint8: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Apply an Albumentations transform to a ``uint8`` RGB image and cast to ``float32`` [0, 255]."""
    if img_uint8.dtype != np.uint8:
        raise TypeError(
            f"Albumentations pipeline requires uint8 input to avoid [0, 255] float32 corruption; got {img_uint8.dtype}."
        )
    augmented = transform(image=img_uint8)["image"]
    return np.clip(augmented, 0, 255).astype(np.float32)


def augment_image_file(
    filepath: str,
    class_name: str,
    img_size: int = 224,
    tier: str = "tier2",
    training: bool = True,
) -> np.ndarray:
    """Load a raw ``uint8`` image, apply class-aware augmentation, and return ``float32`` [0, 255]."""
    img_uint8 = load_image_uint8(filepath, fallback_size=img_size)
    if training:
        transform = build_train_augmentation(
            img_size=img_size,
            tier=tier,
            allow_horizontal_flip=not is_orientation_sensitive(class_name),
        )
    else:
        transform = build_eval_augmentation(img_size=img_size)
    return apply_augmentation_to_float32(img_uint8, transform)

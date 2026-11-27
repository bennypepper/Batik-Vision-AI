"""Headless Colab v4 multi-backbone benchmark, calibration, and ONNX export pipeline.

Trains and compares three modern vision backbones on the leak-free v2.0 Indonesian Batik
benchmark (metadata.csv with DINOv2 cluster_id groups + 177 OOD calibration negatives):
  1. dinov2_vits14   (timm: vit_small_patch14_dinov2.lvd142m, 224x224)
  2. convnext_tiny   (timm: convnext_tiny.fb_in22k_ft_in1k_384, 288x288)
  3. efficientnetv2s (timm: tf_efficientnetv2_s.in21k_ft_in1k, 300x300)

Exports the winning model with built-in NHWC [0..255] float32 -> NCHW ImageNet-normalized
preprocessing directly inside the ONNX graph (opset 17, dynamic batch axis) so it is 100%
drop-in compatible with web/src/lib/inference.worker.ts and web/src/lib/classifier.ts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ORIENTATION_SENSITIVE_CLASSES = frozenset({
    "Solo_Parang",
    "Yogyakarta_Parang",
    "Sulawesi_Selatan_Lontara",
})

KAGGLE_PUBLIC_ZIP_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "fisheightcharacter/indonesian-batik-dataset-enhanced-and-cleaned"
)

BACKBONE_SPECS: dict[str, dict[str, Any]] = {
    "dinov2_vits14": {
        "timm_name": "vit_small_patch14_dinov2.lvd142m",
        "img_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "backbone_lr": 2.5e-5,
        "head_lr": 5e-4,
        "unfreeze_mode": "vit_last6",
    },
    "convnext_tiny": {
        "timm_name": "convnext_tiny.fb_in22k_ft_in1k_384",
        "img_size": 288,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "backbone_lr": 6e-5,
        "head_lr": 5e-4,
        "unfreeze_mode": "convnext_last2",
    },
    "efficientnetv2s": {
        "timm_name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "img_size": 300,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "backbone_lr": 1e-4,
        "head_lr": 5e-4,
        "unfreeze_mode": "effnet_top",
    },
}


def ensure_packages() -> None:
    """Install lightweight missing packages (timm, albumentations, onnxruntime, onnx, kagglehub) on Colab."""
    needed = []
    for mod, pkg in [
        ("timm", "timm"),
        ("albumentations", "albumentations"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("onnxscript", "onnxscript"),
        ("kagglehub", "kagglehub"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        print(f"[setup] Installing missing packages: {needed}", flush=True)
        cmd = ["uv", "pip", "install", "--system", "-q", *needed]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)


def download_kaggle_dataset(data_dir: Path) -> Path:
    """Download & unpack the public v2.0 curated batik dataset via kagglehub if not already present."""
    if (data_dir / "metadata.csv").exists() and (data_dir / "dataset").exists():
        print(f"[data] Using existing dataset at {data_dir}", flush=True)
        return data_dir

    import kagglehub

    t0 = time.time()
    print("[data] Downloading public Kaggle v2.0 dataset via kagglehub...", flush=True)
    hub_dir = Path(kagglehub.dataset_download("fisheightcharacter/indonesian-batik-dataset-enhanced-and-cleaned"))
    print(f"[data] kagglehub ready at {hub_dir} in {time.time() - t0:.1f}s", flush=True)

    # Check if already unpacked with dataset/ + metadata.csv directly inside hub_dir
    if (hub_dir / "metadata.csv").exists() and (hub_dir / "dataset").exists():
        return hub_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    # Symlink or copy top-level files and unpack nested zips into data_dir
    for item in hub_dir.iterdir():
        dst = data_dir / item.name
        if item.is_file() and item.suffix.lower() == ".zip":
            target_dir = data_dir / item.stem
            if not target_dir.exists():
                print(f"[data] Unpacking {item.name} -> {data_dir}...", flush=True)
                with zipfile.ZipFile(item, "r") as zf:
                    first = zf.namelist()[0] if zf.namelist() else ""
                    if first.startswith(item.stem + "/"):
                        zf.extractall(data_dir)
                    else:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        zf.extractall(target_dir)
        elif not dst.exists():
            if item.is_dir():
                os.symlink(item, dst)
            else:
                os.symlink(item, dst)

    assert (data_dir / "metadata.csv").exists(), f"metadata.csv missing in {data_dir} (hub_dir={list(hub_dir.iterdir())})"
    assert (data_dir / "dataset").exists(), f"dataset/ missing in {data_dir}"
    return data_dir


# --- Pure-NumPy Calibration & OOD Utilities (matches src/ml/calibration.py exactly) ---
def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    probs = softmax_with_temperature(logits, temperature=temperature)
    labels_arr = np.asarray(labels, dtype=np.int64)
    true_probs = np.clip(probs[np.arange(len(labels_arr)), labels_arr], 1e-12, 1.0)
    return float(-np.mean(np.log(true_probs)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    grid = np.geomspace(0.1, 10.0, num=80)
    losses = np.array([negative_log_likelihood(logits, labels, float(t)) for t in grid])
    best_idx = int(np.argmin(losses))
    low = float(grid[max(0, best_idx - 1)])
    high = float(grid[min(len(grid) - 1, best_idx + 1)])
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = high - inv_phi * (high - low)
    d = low + inv_phi * (high - low)
    fc = negative_log_likelihood(logits, labels, c)
    fd = negative_log_likelihood(logits, labels, d)
    for _ in range(40):
        if fc < fd:
            high, d, fd = d, c, fc
            c = high - inv_phi * (high - low)
            fc = negative_log_likelihood(logits, labels, c)
        else:
            low, c, fc = c, d, fd
            d = low + inv_phi * (high - low)
            fd = negative_log_likelihood(logits, labels, d)
    return float(0.5 * (low + high))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    confidences = np.max(probs_arr, axis=-1)
    predictions = np.argmax(probs_arr, axis=-1)
    accuracies = (predictions == labels_arr).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels_arr)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(accuracies[mask])) - float(np.mean(confidences[mask])))
    return float(ece)


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1) / math.log(float(p.shape[-1]))


def compute_effective_class_weights(counts: list[int], beta: float = 0.999, max_weight: float = 3.0) -> np.ndarray:
    c = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    eff = 1.0 - np.power(beta, c)
    w = (1.0 - beta) / eff
    w = w / np.mean(w)
    w = np.clip(w, 0.25, max_weight)
    return w.astype(np.float32)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calibrate_ood_thresholds(
    id_probs: np.ndarray,
    ood_probs: np.ndarray,
    max_id_false_reject: float = 0.05,
) -> dict[str, Any]:
    """Sweep (ood_min_confidence, ood_entropy_threshold) to maximize OOD recall at <=5% ID false rejection."""
    from sklearn.metrics import roc_auc_score

    id_msp = np.max(id_probs, axis=1)
    ood_msp = np.max(ood_probs, axis=1)
    id_ent = normalized_entropy(id_probs)
    ood_ent = normalized_entropy(ood_probs)

    y_bin = np.concatenate([np.zeros(len(id_probs)), np.ones(len(ood_probs))])
    # Higher score = more likely OOD
    auroc_msp = float(roc_auc_score(y_bin, np.concatenate([1.0 - id_msp, 1.0 - ood_msp])))
    auroc_ent = float(roc_auc_score(y_bin, np.concatenate([id_ent, ood_ent])))

    best = {
        "ood_min_confidence": 0.40,
        "ood_entropy_threshold": 0.65,
        "ood_recall": 0.0,
        "id_false_reject": 1.0,
        "auroc_msp": round(auroc_msp, 4),
        "auroc_entropy": round(auroc_ent, 4),
    }
    for min_conf in np.linspace(0.20, 0.75, 23):
        for ent_th in np.linspace(0.35, 0.85, 21):
            id_flag = (id_msp < min_conf) | (id_ent > ent_th)
            frr = float(np.mean(id_flag))
            if frr <= max_id_false_reject:
                ood_flag = (ood_msp < min_conf) | (ood_ent > ent_th)
                rec = float(np.mean(ood_flag))
                if rec > best["ood_recall"] or (
                    abs(rec - best["ood_recall"]) < 1e-6 and frr < best["id_false_reject"]
                ):
                    best.update({
                        "ood_min_confidence": round(float(min_conf), 4),
                        "ood_entropy_threshold": round(float(ent_th), 4),
                        "ood_recall": round(rec, 4),
                        "id_false_reject": round(frr, 4),
                    })
    return best


def build_splits(data_dir: Path, seed: int) -> tuple[list[str], list[dict], list[dict], list[dict], list[dict]]:
    """Load metadata.csv and create a strictly leak-free 70/15/15 split grouped by cluster_id."""
    from sklearn.model_selection import StratifiedGroupKFold

    rows = []
    with open(data_dir / "metadata.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = data_dir / r["rel_path"]
            if not p.exists():
                # fallback if extracted without dataset/ prefix
                p = data_dir / "dataset" / r["rel_path"].replace("dataset/", "")
            r["abs_path"] = str(p)
            rows.append(r)

    labels = sorted({r["label"] for r in rows})
    lbl2idx = {c: i for i, c in enumerate(labels)}
    y_all = np.array([lbl2idx[r["label"]] for r in rows], dtype=np.int64)
    g_all = np.array([r["cluster_id"] for r in rows])

    # 7-fold StratifiedGroupKFold: fold 0 = Holdout Test (~14.3%), fold 1 = Val (~14.3%), folds 2..6 = Train (~71.4%)
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    fold_indices = [va for _, va in sgkf.split(np.zeros(len(rows)), y_all, groups=g_all)]
    test_idx = set(fold_indices[0].tolist())
    val_idx = set(fold_indices[1].tolist())
    train_idx = [i for i in range(len(rows)) if i not in test_idx and i not in val_idx]

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in sorted(val_idx)]
    test_rows = [rows[i] for i in sorted(test_idx)]

    # Strictly verify ZERO shared cluster_id across train, val, test!
    c_tr = {r["cluster_id"] for r in train_rows}
    c_va = {r["cluster_id"] for r in val_rows}
    c_te = {r["cluster_id"] for r in test_rows}
    assert len(c_tr & c_va) == 0, "LEAKAGE: shared cluster between train and val!"
    assert len(c_tr & c_te) == 0, "LEAKAGE: shared cluster between train and test!"
    assert len(c_va & c_te) == 0, "LEAKAGE: shared cluster between val and test!"

    ood_rows = []
    ood_csv = data_dir / "ood_negatives" / "ood_manifest.csv"
    if ood_csv.exists():
        with open(ood_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = data_dir / r["rel_path"]
                if not p.exists():
                    p = data_dir / "ood_negatives" / r["rel_path"].replace("ood_negatives/", "")
                if p.exists():
                    r["abs_path"] = str(p)
                    ood_rows.append(r)

    print(
        f"[split] Leak-free grouped split verified (0 shared clusters): "
        f"train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}, ood={len(ood_rows)}",
        flush=True,
    )
    return labels, train_rows, val_rows, test_rows, ood_rows


def run_benchmark(args: argparse.Namespace) -> None:
    ensure_packages()

    import albumentations as A
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import onnx
    import onnxruntime as ort
    import seaborn as sns
    import timm
    import torch
    from PIL import Image, ImageOps
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(
        f"[env] device={device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}), "
        f"amp_dtype={amp_dtype}, torch={torch.__version__}, timm={timm.__version__}",
        flush=True,
    )

    data_dir = download_kaggle_dataset(Path(args.data_dir))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, train_rows, val_rows, test_rows, ood_rows = build_splits(data_dir, seed=args.seed)
    num_classes = len(labels)
    lbl2idx = {c: i for i, c in enumerate(labels)}

    # Optional smoke-test subsample (deterministic, keeps all 28 classes)
    if args.smoke_test:
        def _sub(rlist, per_cls=4):
            by_c = Counter()
            out = []
            for r in rlist:
                if by_c[r["label"]] < per_cls:
                    out.append(r)
                    by_c[r["label"]] += 1
            return out
        train_rows = _sub(train_rows, 4)
        val_rows = _sub(val_rows, 2)
        test_rows = _sub(test_rows, 2)
        ood_rows = ood_rows[:24]
        print(f"[smoke] Subsampled train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}, ood={len(ood_rows)}", flush=True)

    # Dataset wrapper with orientation-safe Albumentations on uint8 RGB
    class BatikDataset(Dataset):
        def __init__(self, rows: list[dict], img_size: int, mean, std, is_train: bool):
            self.rows = rows
            self.is_train = is_train
            self.img_size = img_size
            self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
            self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

            if is_train:
                self.tfm_flip = A.Compose([
                    A.RandomResizedCrop(size=(img_size, img_size), scale=(0.68, 1.0), ratio=(0.85, 1.18), p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.Affine(scale=(0.92, 1.08), rotate=(-20, 20), border_mode=cv2.BORDER_REFLECT_101, p=0.65),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=18, val_shift_limit=18, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.5),
                    A.OneOf([
                        A.ImageCompression(quality_range=(55, 95), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
                    ], p=0.30),
                    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(int(img_size * 0.06), int(img_size * 0.14)),
                                    hole_width_range=(int(img_size * 0.06), int(img_size * 0.14)), p=0.25),
                ])
                # Strict orientation protection: NO horizontal flip, mild rotation <= 12 deg!
                self.tfm_noflip = A.Compose([
                    A.RandomResizedCrop(size=(img_size, img_size), scale=(0.72, 1.0), ratio=(0.88, 1.14), p=1.0),
                    A.Affine(scale=(0.94, 1.06), rotate=(-12, 12), border_mode=cv2.BORDER_REFLECT_101, p=0.60),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=18, val_shift_limit=18, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.5),
                    A.OneOf([
                        A.ImageCompression(quality_range=(55, 95), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
                    ], p=0.30),
                ])
            else:
                self.tfm_eval = A.Compose([
                    A.SmallestMaxSize(max_size=img_size),
                    A.CenterCrop(height=img_size, width=img_size),
                ])

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int):
            r = self.rows[idx]
            with Image.open(r["abs_path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                arr = np.asarray(im, dtype=np.uint8)
            if self.is_train:
                tfm = self.tfm_noflip if r["label"] in ORIENTATION_SENSITIVE_CLASSES else self.tfm_flip
                arr = tfm(image=arr)["image"]
            else:
                arr = self.tfm_eval(image=arr)["image"]
            x = arr.astype(np.float32).transpose(2, 0, 1) / 255.0
            x = (x - self.mean) / self.std
            y = lbl2idx[r["label"]] if "label" in r else -1
            return torch.from_numpy(x), y

    # 3-tile aspect-preserving extractor matching web/src/lib/classifier.ts!
    def extract_web_tiles_nhwc_u8(img_path: str, input_size: int) -> np.ndarray:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            min_s = min(w, h)
            cx, cy = (w - min_s) // 2, (h - min_s) // 2
            rects = [(cx, cy, cx + min_s, cy + min_s)]
            if min_s >= 260:
                ms = max(64, round(min_s * 0.65))
                max_x, max_y = max(0, w - ms), max(0, h - ms)
                if w > h * 1.15:
                    y0 = (h - ms) // 2
                    r1 = (min(max_x, max(0, round(w * 0.25 - ms / 2))), y0)
                    r2 = (min(max_x, max(0, round(w * 0.75 - ms / 2))), y0)
                elif h > w * 1.15:
                    x0 = (w - ms) // 2
                    r1 = (x0, min(max_y, max(0, round(h * 0.25 - ms / 2))))
                    r2 = (x0, min(max_y, max(0, round(h * 0.75 - ms / 2))))
                else:
                    ix, iy = max(0, round((w - ms) * 0.25)), max(0, round((h - ms) * 0.25))
                    r1 = (min(max_x, ix), min(max_y, iy))
                    r2 = (min(max_x, max(0, w - ms - ix)), min(max_y, max(0, h - ms - iy)))
                rects.extend([(r1[0], r1[1], r1[0] + ms, r1[1] + ms), (r2[0], r2[1], r2[0] + ms, r2[1] + ms)])
            tiles = []
            for box in rects:
                crop = im.crop(box).resize((input_size, input_size), Image.Resampling.BILINEAR)
                tiles.append(np.asarray(crop, dtype=np.float32))
            return np.stack(tiles, axis=0)  # [T, H, W, 3] in [0..255]

    class BatikNet(nn.Module):
        def __init__(self, timm_name: str, num_classes: int, img_size: int, pretrained: bool = True):
            super().__init__()
            kwargs = {"pretrained": pretrained, "num_classes": 0}
            if "vit_" in timm_name or "dinov2" in timm_name:
                kwargs["img_size"] = img_size
            self.backbone = timm.create_model(timm_name, **kwargs)
            feat_dim = self.backbone.num_features
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(0.35),
                nn.Linear(feat_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.backbone(x)
            return self.head(feats)

    class WebExportWrapper(nn.Module):
        """Accepts [B, H, W, 3] float32 in [0..255] (exact web worker contract) and outputs [B, 28] logits."""
        def __init__(self, core: BatikNet, mean: tuple[float, float, float], std: tuple[float, float, float]):
            super().__init__()
            self.core = core
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        def forward(self, x_nhwc_255: torch.Tensor) -> torch.Tensor:
            x = x_nhwc_255.permute(0, 3, 1, 2) / 255.0
            x = (x - self.mean) / self.std
            return self.core(x)

    def configure_phase2_unfreeze(model: BatikNet, mode: str) -> int:
        for p in model.backbone.parameters():
            p.requires_grad = False
        if mode == "vit_last6":
            # Unfreeze last 6 transformer blocks + final norm
            blocks = list(model.backbone.blocks)
            for blk in blocks[-6:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for attr in ("norm", "fc_norm"):
                m = getattr(model.backbone, attr, None)
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad = True
        elif mode == "convnext_last2":
            # Unfreeze stages[2], stages[3], norm_pre/head norm
            stages = list(model.backbone.stages)
            for st in stages[-2:]:
                for p in st.parameters():
                    p.requires_grad = True
            for attr in ("norm", "head"):
                m = getattr(model.backbone, attr, None)
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad = True
        elif mode == "effnet_top":
            # Unfreeze top 3 stage blocks + conv_head, keep BatchNorm2d frozen for small-batch stability
            blocks = list(model.backbone.blocks)
            for blk in blocks[-3:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for attr in ("conv_head", "bn2"):
                m = getattr(model.backbone, attr, None)
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad = True
            for m in model.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
        return sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)

    @torch.no_grad()
    def predict_logits(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_logits, all_y = [], []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(xb)
            all_logits.append(logits.float().cpu().numpy())
            all_y.append(yb.numpy())
        return np.concatenate(all_logits, axis=0), np.concatenate(all_y, axis=0)

    tr_counts = [sum(1 for r in train_rows if r["label"] == c) for c in labels]
    cls_weights_t = torch.from_numpy(compute_effective_class_weights(tr_counts, beta=0.999, max_weight=3.0)).to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_weights_t, label_smoothing=0.05)

    backbones_to_run = [b.strip() for b in args.backbones.split(",") if b.strip()]
    benchmark_results: dict[str, Any] = {}
    best_model_name = None
    best_model_score = -1.0

    p1_epochs = 1 if args.smoke_test else args.phase1_epochs
    p2_epochs = 1 if args.smoke_test else args.phase2_epochs
    num_workers = 2 if os.name != "nt" else 0

    for bname in backbones_to_run:
        spec = BACKBONE_SPECS[bname]
        img_size = spec["img_size"]
        print(f"\n{'=' * 85}\n[TRAIN] Backbone={bname} ({spec['timm_name']}, {img_size}x{img_size})\n{'=' * 85}", flush=True)
        t_start = time.time()

        tr_ds = BatikDataset(train_rows, img_size, spec["mean"], spec["std"], is_train=True)
        va_ds = BatikDataset(val_rows, img_size, spec["mean"], spec["std"], is_train=False)
        te_ds = BatikDataset(test_rows, img_size, spec["mean"], spec["std"], is_train=False)
        oo_ds = BatikDataset(ood_rows, img_size, spec["mean"], spec["std"], is_train=False)

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        oo_loader = DataLoader(oo_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        model = BatikNet(spec["timm_name"], num_classes=num_classes, img_size=img_size, pretrained=not args.smoke_test).to(device)
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and amp_dtype == torch.float16))

        # Phase 1: Frozen backbone head warmup
        for p in model.backbone.parameters():
            p.requires_grad = False
        opt1 = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-2)
        best_val_f1 = -1.0
        ckpt_path = out_dir / f"{bname}_best.pt"

        for ep in range(1, p1_epochs + 1):
            model.train()
            model.backbone.eval()
            tr_loss = 0.0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt1.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt1)
                scaler.update()
                tr_loss += float(loss.item()) * len(yb)
            va_logits, va_y = predict_logits(model, va_loader)
            va_pred = va_logits.argmax(axis=1)
            va_acc = float(accuracy_score(va_y, va_pred))
            va_f1 = float(f1_score(va_y, va_pred, average="macro", zero_division=0))
            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                torch.save(model.state_dict(), ckpt_path)
            print(f"  [P1 {ep:02d}/{p1_epochs}] loss={tr_loss/len(tr_ds):.4f} | val_acc={va_acc*100:.2f}% | val_macro_f1={va_f1*100:.2f}%", flush=True)

        # Phase 2: Partial fine-tuning with discriminative LR + Cosine Warmup
        n_unfrozen = configure_phase2_unfreeze(model, spec["unfreeze_mode"])
        print(f"  [P2] Unfroze {n_unfrozen/1e6:.2f}M backbone params (mode={spec['unfreeze_mode']})", flush=True)
        bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
        hd_params = list(model.head.parameters())
        opt2 = torch.optim.AdamW([
            {"params": bb_params, "lr": spec["backbone_lr"], "weight_decay": 0.05},
            {"params": hd_params, "lr": spec["head_lr"], "weight_decay": 0.02},
        ])
        total_steps = max(1, len(tr_loader) * p2_epochs)
        warmup_steps = max(1, int(0.10 * total_steps))

        def lr_lambda(step: int, _ws: int = warmup_steps, _ts: int = total_steps) -> float:
            if step < _ws:
                return float(step + 1) / float(_ws)
            prog = (step - _ws) / max(1, _ts - _ws)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * prog)))

        sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda)
        no_improve = 0
        patience = 7

        for ep in range(1, p2_epochs + 1):
            model.train()
            if spec["unfreeze_mode"] == "effnet_top":
                for m in model.backbone.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
            tr_loss = 0.0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt2.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
                scaler.step(opt2)
                scaler.update()
                if not scaler.is_enabled() or scaler.get_scale() >= old_scale:
                    sched2.step()
                tr_loss += float(loss.item()) * len(yb)

            va_logits, va_y = predict_logits(model, va_loader)
            va_pred = va_logits.argmax(axis=1)
            va_acc = float(accuracy_score(va_y, va_pred))
            va_f1 = float(f1_score(va_y, va_pred, average="macro", zero_division=0))
            improved = ""
            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
                improved = " *BEST*"
            else:
                no_improve += 1
            print(f"  [P2 {ep:02d}/{p2_epochs}] loss={tr_loss/len(tr_ds):.4f} | val_acc={va_acc*100:.2f}% | val_macro_f1={va_f1*100:.2f}%{improved}", flush=True)
            if no_improve >= patience:
                print(f"  [P2] Early stopping after {ep} epochs (best val_macro_f1={best_val_f1*100:.2f}%)", flush=True)
                break

        # Reload best checkpoint & run calibration + holdout test + OOD evaluation
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        va_logits, va_y = predict_logits(model, va_loader)
        te_logits, te_y = predict_logits(model, te_loader)
        oo_logits, _ = predict_logits(model, oo_loader)

        best_T = fit_temperature(va_logits, va_y)
        va_probs_raw = softmax_with_temperature(va_logits, 1.0)
        va_probs_cal = softmax_with_temperature(va_logits, best_T)
        te_probs_raw = softmax_with_temperature(te_logits, 1.0)
        te_probs_cal = softmax_with_temperature(te_logits, best_T)
        oo_probs_cal = softmax_with_temperature(oo_logits, best_T)

        val_ece_before = compute_ece(va_probs_raw, va_y)
        val_ece_after = compute_ece(va_probs_cal, va_y)
        test_ece_before = compute_ece(te_probs_raw, te_y)
        test_ece_after = compute_ece(te_probs_cal, te_y)

        te_pred = te_probs_cal.argmax(axis=1)
        te_top1 = float(accuracy_score(te_y, te_pred))
        te_top3 = float(np.mean([te_y[i] in np.argsort(-te_probs_cal[i])[:3] for i in range(len(te_y))]))
        te_f1 = float(f1_score(te_y, te_pred, average="macro", zero_division=0))

        ood_cal = calibrate_ood_thresholds(va_probs_cal, oo_probs_cal, max_id_false_reject=0.05)
        # Measure OOD rejection on test batik + OOD negatives + specifically Indonesian non-batik textiles!
        te_msp = np.max(te_probs_cal, axis=1)
        te_ent = normalized_entropy(te_probs_cal)
        te_frr = float(np.mean((te_msp < ood_cal["ood_min_confidence"]) | (te_ent > ood_cal["ood_entropy_threshold"])))

        oo_msp = np.max(oo_probs_cal, axis=1)
        oo_ent = normalized_entropy(oo_probs_cal)
        oo_flags = (oo_msp < ood_cal["ood_min_confidence"]) | (oo_ent > ood_cal["ood_entropy_threshold"])
        ood_by_bucket = {}
        for bkt in sorted({r["bucket"] for r in ood_rows}):
            b_idx = [i for i, r in enumerate(ood_rows) if r["bucket"] == bkt]
            ood_by_bucket[bkt] = {"count": len(b_idx), "rejected_rate": round(float(np.mean(oo_flags[b_idx])), 4)}

        elapsed = round(time.time() - t_start, 1)
        res_entry = {
            "backbone": bname,
            "timm_name": spec["timm_name"],
            "img_size": img_size,
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "train_time_sec": elapsed,
            "best_val_macro_f1": round(best_val_f1, 4),
            "test_top1_acc": round(te_top1, 4),
            "test_top3_acc": round(te_top3, 4),
            "test_macro_f1": round(te_f1, 4),
            "temperature": round(best_T, 4),
            "val_ece_before": round(val_ece_before, 4),
            "val_ece_after": round(val_ece_after, 4),
            "test_ece_before": round(test_ece_before, 4),
            "test_ece_after": round(test_ece_after, 4),
            "ood_calibration": ood_cal,
            "test_id_false_reject_rate": round(te_frr, 4),
            "ood_by_bucket": ood_by_bucket,
        }
        benchmark_results[bname] = res_entry
        print(
            f"[RESULT] {bname}: Val Macro-F1={best_val_f1*100:.2f}% | "
            f"Test Top1={te_top1*100:.2f}% | Top3={te_top3*100:.2f}% | Test Macro-F1={te_f1*100:.2f}% | "
            f"T={best_T:.3f} | Test ECE={test_ece_before:.4f}->{test_ece_after:.4f} | "
            f"OOD AUROC={ood_cal['auroc_msp']:.4f} | OOD Recall={ood_cal['ood_recall']*100:.1f}% (Test FRR={te_frr*100:.1f}%) | {elapsed}s",
            flush=True,
        )

        # Select winning backbone strictly by Val Macro-F1 (no peeking at test set for model selection!)
        if best_val_f1 > best_model_score:
            best_model_score = best_val_f1
            best_model_name = bname

        del model, tr_ds, va_ds, te_ds, oo_ds, tr_loader, va_loader, te_loader, oo_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Export Winner -> ONNX (FP32, FP16, INT8) + Exact Web Parity & 3-Tile Test Eval
    # ------------------------------------------------------------------
    assert best_model_name is not None
    win_spec = BACKBONE_SPECS[best_model_name]
    win_size = win_spec["img_size"]
    win_res = benchmark_results[best_model_name]
    print(f"\n{'=' * 85}\n[EXPORT WINNER] {best_model_name} (Val Macro-F1={best_model_score*100:.2f}%, {win_size}x{win_size})\n{'=' * 85}", flush=True)

    win_core = BatikNet(win_spec["timm_name"], num_classes=num_classes, img_size=win_size, pretrained=False)
    win_core.load_state_dict(torch.load(out_dir / f"{best_model_name}_best.pt", map_location="cpu", weights_only=True))
    win_core.eval()
    wrapper = WebExportWrapper(win_core, win_spec["mean"], win_spec["std"]).eval().cpu()

    fp32_path = out_dir / "batik_model_v4_fp32.onnx"
    dummy_in = torch.rand(1, win_size, win_size, 3, dtype=torch.float32) * 255.0
    print(f"[export] Exporting FP32 ONNX (opset 17, NHWC [B,{win_size},{win_size},3] -> [B,{num_classes}])...", flush=True)
    torch.onnx.export(
        wrapper,
        (dummy_in,),
        str(fp32_path),
        input_names=["input_image"],
        output_names=["logits"],
        dynamic_axes={"input_image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    # Inline any external tensor data into a single self-contained .onnx file
    onnx_model = onnx.load(str(fp32_path), load_external_data=True)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(fp32_path), save_as_external_data=False)
    # Remove any leftover .data sidecar
    for sidecar in out_dir.glob("*.onnx.data"):
        sidecar.unlink(missing_ok=True)
    print(f"  FP32 size: {fp32_path.stat().st_size / 1e6:.2f} MB", flush=True)

    # Dynamic INT8 quantization (weights quantized to QInt8/QUInt8 — works natively on onnxruntime-web WASM!)
    int8_path = out_dir / "batik_model_v4_int8.onnx"
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
    )
    print(f"  INT8 size: {int8_path.stat().st_size / 1e6:.2f} MB", flush=True)

    # FP16 conversion (keep FP32 I/O for drop-in JS Float32Array compatibility)
    fp16_path = out_dir / "batik_model_v4_fp16.onnx"
    try:
        from onnxruntime.transformers import float16 as ort_fp16
        m_fp16 = ort_fp16.convert_float_to_float16(onnx.load(str(fp32_path)), keep_io_types=True)
        onnx.save(m_fp16, str(fp16_path))
        print(f"  FP16 size: {fp16_path.stat().st_size / 1e6:.2f} MB", flush=True)
    except (ImportError, RuntimeError, ValueError, OSError) as e:
        print(f"  [warn] FP16 conversion skipped: {e}", flush=True)
        fp16_path = None

    # Verify ONNX Parity at batch_size=1 and batch_size=3 (web multi-patch contract!)
    parity_report: dict[str, Any] = {}
    for tag, mpath in [("fp32", fp32_path), ("int8", int8_path), ("fp16", fp16_path)]:
        if mpath is None or not mpath.exists():
            continue
        try:
            sess = ort.InferenceSession(str(mpath), providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            b_stats = {}
            for bs in (1, 3):
                rng = np.random.default_rng(42 + bs)
                x_np = (rng.random((bs, win_size, win_size, 3), dtype=np.float32) * 255.0)
                with torch.no_grad():
                    pt_out = wrapper(torch.from_numpy(x_np)).numpy()
                t0_inf = time.perf_counter()
                ort_out = sess.run(None, {in_name: x_np})[0]
                ms = (time.perf_counter() - t0_inf) * 1000.0
                max_diff = float(np.max(np.abs(pt_out - ort_out)))
                top1_match = bool(np.all(pt_out.argmax(axis=1) == ort_out.argmax(axis=1)))
                b_stats[f"batch_{bs}"] = {"max_abs_diff": round(max_diff, 5), "top1_match": top1_match, "cpu_ms": round(ms, 1)}
            parity_report[tag] = {"size_mb": round(mpath.stat().st_size / 1e6, 2), **b_stats}
            print(f"  [parity {tag}] size={parity_report[tag]['size_mb']} MB | b1={b_stats['batch_1']} | b3={b_stats['batch_3']}", flush=True)
        except (RuntimeError, ValueError, OSError) as e:
            parity_report[tag] = {"error": str(e)}
            print(f"  [parity {tag}] FAILED: {e}", flush=True)

    # Full Holdout Test Evaluation using GPU FP32 3-tile + ACTUAL INT8 ONNX 3-tile web voting!
    T_win = win_res["temperature"]

    @torch.no_grad()
    def eval_torch_web_multitile(wrap_mod: nn.Module, eval_rows: list[dict], has_labels: bool = True) -> dict[str, Any]:
        wrap_mod = wrap_mod.to(device).eval()
        probs_list = []
        y_list = []
        for r in eval_rows:
            tiles = extract_web_tiles_nhwc_u8(r["abs_path"], win_size)
            xt = torch.from_numpy(tiles).to(device)
            logits = wrap_mod(xt).float().cpu().numpy()
            tile_probs = softmax_with_temperature(logits, T_win)
            h_norms = normalized_entropy(tile_probs)
            weights = np.maximum(0.05, 1.0 - h_norms)
            agg = np.sum(tile_probs * weights[:, None], axis=0) / np.sum(weights)
            probs_list.append(agg)
            if has_labels:
                y_list.append(lbl2idx[r["label"]])
        wrap_mod.cpu()
        probs_arr = np.stack(probs_list, axis=0)
        if not has_labels:
            return {"probs": probs_arr}
        y_arr = np.array(y_list, dtype=np.int64)
        pred = probs_arr.argmax(axis=1)
        return {
            "probs": probs_arr,
            "y_true": y_arr,
            "y_pred": pred,
            "top1_acc": round(float(accuracy_score(y_arr, pred)), 4),
            "top3_acc": round(float(np.mean([y_arr[i] in np.argsort(-probs_arr[i])[:3] for i in range(len(y_arr))])), 4),
            "macro_f1": round(float(f1_score(y_arr, pred, average="macro", zero_division=0)), 4),
            "ece": round(compute_ece(probs_arr, y_arr), 4),
        }

    def eval_onnx_web_multitile(mpath: Path, eval_rows: list[dict], has_labels: bool = True) -> dict[str, Any]:
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, os.cpu_count() or 4)
        sess = ort.InferenceSession(str(mpath), sess_options=so, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        probs_list = []
        y_list = []
        for r in eval_rows:
            tiles = extract_web_tiles_nhwc_u8(r["abs_path"], win_size)
            logits = sess.run(None, {in_name: tiles})[0]
            tile_probs = softmax_with_temperature(logits, T_win)
            h_norms = normalized_entropy(tile_probs)
            weights = np.maximum(0.05, 1.0 - h_norms)
            agg = np.sum(tile_probs * weights[:, None], axis=0) / np.sum(weights)
            probs_list.append(agg)
            if has_labels:
                y_list.append(lbl2idx[r["label"]])
        probs_arr = np.stack(probs_list, axis=0)
        if not has_labels:
            return {"probs": probs_arr}
        y_arr = np.array(y_list, dtype=np.int64)
        pred = probs_arr.argmax(axis=1)
        return {
            "probs": probs_arr,
            "y_true": y_arr,
            "y_pred": pred,
            "top1_acc": round(float(accuracy_score(y_arr, pred)), 4),
            "top3_acc": round(float(np.mean([y_arr[i] in np.argsort(-probs_arr[i])[:3] for i in range(len(y_arr))])), 4),
            "macro_f1": round(float(f1_score(y_arr, pred, average="macro", zero_division=0)), 4),
            "ece": round(compute_ece(probs_arr, y_arr), 4),
        }

    # Evaluate FP32 (via exact GPU wrapper) + INT8 ONNX on holdout test & OOD (exact 3-tile web voting!)
    print("[eval] Evaluating FP32 (GPU) & INT8 ONNX with exact web 3-tile voting on holdout test + OOD...", flush=True)
    fp32_web = eval_torch_web_multitile(wrapper, test_rows, has_labels=True)
    fp32_val_web = eval_torch_web_multitile(wrapper, val_rows, has_labels=True)
    fp32_ood_web = eval_torch_web_multitile(wrapper, ood_rows, has_labels=False)
    fp32_ood_cal = calibrate_ood_thresholds(fp32_val_web["probs"], fp32_ood_web["probs"], max_id_false_reject=0.05)

    int8_web = eval_onnx_web_multitile(int8_path, test_rows, has_labels=True)
    int8_ood = eval_onnx_web_multitile(int8_path, ood_rows, has_labels=False)
    print(f"  FP32 (3-tile Test): Top1={fp32_web['top1_acc']*100:.2f}% | Top3={fp32_web['top3_acc']*100:.2f}% | Macro-F1={fp32_web['macro_f1']*100:.2f}% | ECE={fp32_web['ece']:.4f}", flush=True)
    print(f"  INT8 (3-tile Test): Top1={int8_web['top1_acc']*100:.2f}% | Top3={int8_web['top3_acc']*100:.2f}% | Macro-F1={int8_web['macro_f1']*100:.2f}% | ECE={int8_web['ece']:.4f}", flush=True)

    int8_val_web = eval_onnx_web_multitile(int8_path, val_rows, has_labels=True)
    int8_ood_cal = calibrate_ood_thresholds(int8_val_web["probs"], int8_ood["probs"], max_id_false_reject=0.05)
    print(f"  INT8 (3-tile OOD) : AUROC={int8_ood_cal['auroc_msp']:.4f} | Recall={int8_ood_cal['ood_recall']*100:.1f}% | Val FRR={int8_ood_cal['id_false_reject']*100:.1f}%", flush=True)

    # Choose best deployable artifact: INT8 if Macro-F1 drop <= 0.015 (1.5%), else FP16!
    int8_drop = fp32_web["macro_f1"] - int8_web["macro_f1"]
    if int8_drop <= 0.015:
        deploy_tag = "int8"
        deploy_src = int8_path
        deploy_eval = int8_web
        deploy_ood = int8_ood_cal
    elif fp16_path is not None and fp16_path.exists():
        deploy_tag = "fp16"
        deploy_src = fp16_path
        deploy_eval = fp32_web
        deploy_ood = fp32_ood_cal
    else:
        deploy_tag = "fp32"
        deploy_src = fp32_path
        deploy_eval = fp32_web
        deploy_ood = fp32_ood_cal

    final_onnx = out_dir / "batik_model_v4.onnx"
    import shutil
    shutil.copy2(deploy_src, final_onnx)
    print(f"[deploy] Selected {deploy_tag.upper()} -> {final_onnx} ({final_onnx.stat().st_size / 1e6:.2f} MB, drop vs FP32={int8_drop*100:.2f}%)", flush=True)

    # Write model_config.json matching web/src/lib/inference.worker.ts contract
    model_cfg = {
        "version": "4.0.0",
        "backbone": best_model_name,
        "timm_name": win_spec["timm_name"],
        "quantization": deploy_tag,
        "input_size": win_size,
        "temperature": round(T_win, 6),
        "ood_entropy_threshold": round(deploy_ood["ood_entropy_threshold"], 6),
        "ood_min_confidence": round(deploy_ood["ood_min_confidence"], 6),
        "num_classes": num_classes,
        "labels": labels,
    }
    (out_dir / "model_config.json").write_text(json.dumps(model_cfg, indent=2) + "\n", encoding="utf-8")

    # Classification report & confusion matrix PNG on holdout test (deployed ONNX 3-tile!)
    report_txt = classification_report(deploy_eval["y_true"], deploy_eval["y_pred"], target_names=labels, digits=4, zero_division=0)
    (out_dir / "v4_classification_report.txt").write_text(report_txt, encoding="utf-8")

    cm = confusion_matrix(deploy_eval["y_true"], deploy_eval["y_pred"])
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Batik Vision AI v4 Holdout Confusion Matrix ({best_model_name} {deploy_tag.upper()}, 3-Tile Web Eval)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "v4_confusion_matrix.png", dpi=150)
    plt.close()

    # Save complete benchmark summary JSON
    summary_all = {
        "smoke_test": bool(args.smoke_test),
        "seed": args.seed,
        "split_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows), "ood": len(ood_rows)},
        "winner_backbone": best_model_name,
        "deployed_quantization": deploy_tag,
        "deployed_onnx_size_mb": round(final_onnx.stat().st_size / 1e6, 2),
        "backbones": benchmark_results,
        "onnx_parity": parity_report,
        "onnx_web_3tile_eval": {
            "fp32": {k: v for k, v in fp32_web.items() if k not in ("probs", "y_true", "y_pred")},
            "int8": {k: v for k, v in int8_web.items() if k not in ("probs", "y_true", "y_pred")},
            "deployed_ood_calibration": deploy_ood,
        },
        "model_config": model_cfg,
    }
    (out_dir / "v4_benchmark_summary.json").write_text(json.dumps(summary_all, indent=2) + "\n", encoding="utf-8")
    print(f"\n[DONE] Saved all v4 benchmark artifacts to {out_dir}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Colab v4 multi-backbone Batik benchmark & ONNX export")
    p.add_argument("--data-dir", type=str, default="/content/batik_v2")
    p.add_argument("--output-dir", type=str, default="/content/v4_benchmark_out")
    p.add_argument("--backbones", type=str, default="dinov2_vits14,convnext_tiny,efficientnetv2s")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--phase1-epochs", type=int, default=6)
    p.add_argument("--phase2-epochs", type=int, default=25)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args, _ = p.parse_known_args(argv)
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())

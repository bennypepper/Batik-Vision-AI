"""Post-hoc temperature scaling, calibration metrics, and OOD detection (pure NumPy)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable softmax probabilities scaled by temperature T > 0."""
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}.")
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def negative_log_likelihood(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float = 1.0,
) -> float:
    """Compute mean cross-entropy (NLL) of temperature-scaled logits against integer labels."""
    probs = softmax_with_temperature(logits, temperature=temperature)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.ndim == 2:
        labels_arr = np.argmax(labels_arr, axis=-1)
    eps = 1e-12
    true_probs = np.clip(probs[np.arange(len(labels_arr)), labels_arr], eps, 1.0)
    return float(-np.mean(np.log(true_probs)))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float = 0.1,
    t_max: float = 10.0,
    n_grid: int = 80,
    n_iter: int = 40,
) -> float:
    """Fit scalar temperature T > 0 minimizing validation NLL via log-grid + golden-section search."""
    if t_min <= 0.0 or t_max <= t_min:
        raise ValueError(f"Invalid temperature search bounds: [{t_min}, {t_max}].")

    grid = np.geomspace(t_min, t_max, num=n_grid)
    losses = np.array([negative_log_likelihood(logits, labels, float(t)) for t in grid])
    best_idx = int(np.argmin(losses))

    low = float(grid[max(0, best_idx - 1)])
    high = float(grid[min(len(grid) - 1, best_idx + 1)])

    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = high - inv_phi * (high - low)
    d = low + inv_phi * (high - low)
    fc = negative_log_likelihood(logits, labels, c)
    fd = negative_log_likelihood(logits, labels, d)

    for _ in range(n_iter):
        if fc < fd:
            high = d
            d = c
            fd = fc
            c = high - inv_phi * (high - low)
            fc = negative_log_likelihood(logits, labels, c)
        else:
            low = c
            c = d
            fc = fd
            d = low + inv_phi * (high - low)
            fd = negative_log_likelihood(logits, labels, d)

    return float(0.5 * (low + high))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) over ``n_bins`` uniform confidence bins."""
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.ndim == 2:
        labels_arr = np.argmax(labels_arr, axis=-1)

    confidences = np.max(probs_arr, axis=-1)
    predictions = np.argmax(probs_arr, axis=-1)
    accuracies = (predictions == labels_arr).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(labels_arr)
    if n_total == 0:
        return 0.0

    for idx in range(n_bins):
        low, high = bin_edges[idx], bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        bin_count = int(np.sum(mask))
        if bin_count > 0:
            avg_acc = float(np.mean(accuracies[mask]))
            avg_conf = float(np.mean(confidences[mask]))
            ece += (bin_count / n_total) * abs(avg_acc - avg_conf)

    return float(ece)


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy normalized to [0, 1] by ``log(num_classes)``."""
    probs_arr = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    num_classes = probs_arr.shape[-1]
    if num_classes <= 1:
        return np.zeros(probs_arr.shape[:-1], dtype=np.float64)
    entropy = -np.sum(probs_arr * np.log(probs_arr), axis=-1)
    return entropy / np.log(float(num_classes))


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute Helmholtz free energy OOD score: ``-T * log(sum(exp(logits / T)))``."""
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}.")
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    max_val = np.max(scaled, axis=-1, keepdims=True)
    log_sum_exp = np.squeeze(max_val, axis=-1) + np.log(np.sum(np.exp(scaled - max_val), axis=-1))
    return -float(temperature) * log_sum_exp


class OODDetector:
    """Calibrated out-of-distribution detector combining Max-Prob, Entropy, and Energy."""

    def __init__(
        self,
        temperature: float = 1.0,
        entropy_threshold: float = 0.65,
        min_confidence: float = 0.40,
        energy_threshold: float | None = None,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got {temperature}.")
        self.temperature = float(temperature)
        self.entropy_threshold = float(entropy_threshold)
        self.min_confidence = float(min_confidence)
        self.energy_threshold = float(energy_threshold) if energy_threshold is not None else None

    def score(self, logits: np.ndarray) -> dict[str, np.ndarray]:
        """Score a batch of logits and flag out-of-distribution samples."""
        logits_arr = np.asarray(logits, dtype=np.float64)
        if logits_arr.ndim == 1:
            logits_arr = logits_arr[np.newaxis, :]

        probs = softmax_with_temperature(logits_arr, temperature=self.temperature)
        max_prob = np.max(probs, axis=-1)
        norm_ent = normalized_entropy(probs)
        energy = energy_score(logits_arr, temperature=self.temperature)

        is_ood = (max_prob < self.min_confidence) | (norm_ent > self.entropy_threshold)
        if self.energy_threshold is not None:
            is_ood = is_ood | (energy > self.energy_threshold)

        return {
            "probs": probs,
            "max_prob": max_prob,
            "normalized_entropy": norm_ent,
            "energy": energy,
            "is_ood": is_ood,
        }

    def export_model_config(
        self,
        output_path: str | Path,
        labels: Sequence[str],
        version: str = "4.0.0",
        input_size: int = 224,
    ) -> dict[str, Any]:
        """Export JSON matching the web runtime ``model_config.json`` contract."""
        config: dict[str, Any] = {
            "version": version,
            "input_size": int(input_size),
            "temperature": round(self.temperature, 6),
            "ood_entropy_threshold": round(self.entropy_threshold, 6),
            "ood_min_confidence": round(self.min_confidence, 6),
            "num_classes": len(labels),
            "labels": list(labels),
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return config

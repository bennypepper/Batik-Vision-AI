"""Model architecture, class weighting, and learning-rate schedules for Batik Motifs v3/v4."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import tensorflow as tf


def create_v3_backbone_model(
    num_classes: int = 28,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str | None = None,
    img_size: int | None = None,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Create the EfficientNetV2S classifier with single explicit Rescaling layer.

    The model accepts raw RGB float32 values in [0, 255] and rescales internally
    via ``Rescaling(1/127.5, offset=-1.0)``. ``EfficientNetV2S`` is instantiated
    with ``include_preprocessing=False`` to prevent double normalization.

    Args:
        num_classes: Number of target classification categories. Default: 28.
        input_shape: Input tensor shape ``(height, width, channels)``.
        weights: Pretrained weights (``"imagenet"`` or ``None``).
        img_size: Optional square spatial size (e.g. 224, 300, 384). Overrides
            ``input_shape`` when provided.
    """
    resolved_shape = (img_size, img_size, 3) if img_size is not None else input_shape
    inputs = tf.keras.Input(shape=resolved_shape, name="input_image", dtype=tf.float32)

    rescaled = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="input_rescaling",
    )(inputs)

    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights=weights,
        include_preprocessing=False,
        input_shape=resolved_shape,
    )

    features = base_model(rescaled)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(features)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(0.5, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="batik_efficientnetv2s_v3")
    return base_model, model


def compute_effective_class_weights(
    class_counts: Sequence[int],
    beta: float = 0.999,
    max_weight: float | None = 3.0,
) -> np.ndarray:
    """Compute capped class weights based on Effective Number of Samples (Cui et al., CVPR 2019).

    W_c = (1 - beta) / (1 - beta^{N_c}), normalized to mean 1.0 and capped at
    ``max_weight`` so minority classes are boosted without collapsing calibration.
    """
    counts = np.array(class_counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    effective_num = np.where(effective_num <= 0.0, 1e-8, effective_num)

    weights = (1.0 - beta) / effective_num
    normalized_weights = weights / np.sum(weights) * len(counts)
    if max_weight is not None:
        normalized_weights = np.minimum(normalized_weights, float(max_weight))
    return normalized_weights.astype(np.float32)


def compute_effective_sample_weights(
    class_counts: Sequence[int],
    beta: float = 0.999,
    max_weight: float | None = 3.0,
) -> np.ndarray:
    """Backward-compatible alias for ``compute_effective_class_weights``."""
    return compute_effective_class_weights(class_counts, beta=beta, max_weight=max_weight)


def configure_phase2_unfreeze(
    base_model: tf.keras.Model,
    unfreeze_top_n: int = 120,
) -> int:
    """Unfreeze the top ``unfreeze_top_n`` backbone layers while keeping all BatchNorm layers frozen.

    Returns the number of trainable layers in ``base_model``.
    """
    base_model.trainable = True
    cutoff = max(0, len(base_model.layers) - int(unfreeze_top_n))
    for layer in base_model.layers[:cutoff]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return sum(1 for layer in base_model.layers if layer.trainable)


@tf.keras.utils.register_keras_serializable(package="batik_v4")
class CosineWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay down to ``min_lr``."""

    def __init__(
        self,
        initial_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 1e-7,
    ) -> None:
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, min(int(warmup_steps), self.total_steps))
        self.min_lr = float(min_lr)

    def __call__(self, step: tf.Tensor | float) -> tf.Tensor:
        step_f = tf.cast(step, tf.float32)
        warmup_f = tf.cast(self.warmup_steps, tf.float32)
        total_f = tf.cast(self.total_steps, tf.float32)
        init_lr = tf.cast(self.initial_lr, tf.float32)
        min_lr = tf.cast(self.min_lr, tf.float32)

        # Linear warmup: min_lr -> initial_lr
        warmup_frac = tf.cond(
            warmup_f > 0.0,
            lambda: tf.minimum(step_f / tf.maximum(warmup_f, 1.0), 1.0),
            lambda: tf.constant(1.0, dtype=tf.float32),
        )
        warmup_lr = min_lr + (init_lr - min_lr) * warmup_frac

        # Cosine decay: initial_lr -> min_lr
        decay_steps = tf.maximum(total_f - warmup_f, 1.0)
        decay_progress = tf.clip_by_value((step_f - warmup_f) / decay_steps, 0.0, 1.0)
        cosine_factor = 0.5 * (1.0 + tf.cos(tf.constant(math.pi, dtype=tf.float32) * decay_progress))
        decay_lr = min_lr + (init_lr - min_lr) * cosine_factor

        return tf.where(step_f < warmup_f, warmup_lr, decay_lr)

    def get_config(self) -> dict[str, Any]:
        return {
            "initial_lr": self.initial_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


def build_cosine_warmup_schedule(
    initial_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 1e-7,
) -> CosineWarmupSchedule:
    """Create a Keras-serializable cosine decay schedule with linear warmup."""
    return CosineWarmupSchedule(
        initial_lr=initial_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
    )

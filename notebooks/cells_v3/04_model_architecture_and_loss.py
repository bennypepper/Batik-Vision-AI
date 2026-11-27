"""Cell 4: Backbone model architecture, calibrated loss, capped class weights, and cosine warmup."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="batik_v3")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Categorical Focal Loss using true-class probability p_t for modulation."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Any = None,
        label_smoothing: float = 0.0,
        reduction: str = "sum_over_batch_size",
        name: str = "categorical_focal_loss",
    ):
        super().__init__(reduction=reduction, name=name)
        self.gamma = float(gamma)
        self.alpha = alpha
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_hard = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        p_t = tf.reduce_sum(y_true_hard * y_pred, axis=-1, keepdims=True)
        p_t = tf.clip_by_value(p_t, epsilon, 1.0 - epsilon)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        y_target = y_true_hard
        if self.label_smoothing > 0.0:
            num_classes = tf.cast(tf.shape(y_true_hard)[-1], tf.float32)
            y_target = y_true_hard * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        cross_entropy = -y_target * tf.math.log(y_pred)
        if self.alpha is not None:
            cross_entropy = tf.cast(self.alpha, tf.float32) * cross_entropy

        return tf.reduce_sum(modulating_factor * cross_entropy, axis=-1)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
        })
        return config


def build_classification_loss(
    mode: str = "cce_smooth",
    label_smoothing: float = 0.05,
    gamma: float = 2.0,
    alpha: Any = None,
) -> tf.keras.losses.Loss:
    """Build v3/v4 classification loss ('cce_smooth' default for calibrated probabilities)."""
    if mode == "cce_smooth":
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=float(label_smoothing))
    if mode == "focal":
        return CategoricalFocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)
    raise ValueError(f"Unsupported loss mode: {mode!r}")


def compute_effective_sample_weights(
    class_counts: Sequence[int],
    beta: float = 0.999,
    max_weight: float | None = 3.0,
) -> np.ndarray:
    """Compute capped Effective Number of Samples class weights."""
    counts = np.array(class_counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    effective_num = np.where(effective_num <= 0.0, 1e-8, effective_num)

    weights = (1.0 - beta) / effective_num
    normalized_weights = weights / np.sum(weights) * len(counts)
    if max_weight is not None:
        normalized_weights = np.minimum(normalized_weights, float(max_weight))
    return normalized_weights.astype(np.float32)


@tf.keras.utils.register_keras_serializable(package="batik_v4")
class CosineWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay down to min_lr."""

    def __init__(self, initial_lr: float, total_steps: int, warmup_steps: int, min_lr: float = 1e-7):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, min(int(warmup_steps), self.total_steps))
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        warmup_f = tf.cast(self.warmup_steps, tf.float32)
        total_f = tf.cast(self.total_steps, tf.float32)
        init_lr = tf.cast(self.initial_lr, tf.float32)
        min_lr = tf.cast(self.min_lr, tf.float32)

        warmup_frac = tf.cond(
            warmup_f > 0.0,
            lambda: tf.minimum(step_f / tf.maximum(warmup_f, 1.0), 1.0),
            lambda: tf.constant(1.0, dtype=tf.float32),
        )
        warmup_lr = min_lr + (init_lr - min_lr) * warmup_frac

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


def create_v3_backbone_model(
    num_classes: int = 28,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str | None = "imagenet",
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build EfficientNetV2S classifier with single explicit Rescaling layer."""
    inputs = tf.keras.Input(shape=input_shape, name="input_image", dtype=tf.float32)

    rescaled = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="input_rescaling",
    )(inputs)

    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights=weights,
        include_preprocessing=False,
        input_shape=input_shape,
    )

    features = base_model(rescaled)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(features)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE_1, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE_2, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="batik_efficientnetv2s_v3")
    return base_model, model


print("Model architecture, calibrated loss, and cosine warmup schedule ready.")

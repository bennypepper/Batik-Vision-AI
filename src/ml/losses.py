"""Classification losses for Batik Motifs v3/v4 training."""

from __future__ import annotations

from typing import Any

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="batik_v3")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Categorical Focal Loss using true-class probability ``p_t`` for modulation.

    Formulation:
        p_t = sum(y_true_hard * y_pred, axis=-1)
        FL(p_t) = (1 - p_t)^gamma * (-sum(alpha * y_true_smooth * log(y_pred)))

    Computing ``p_t`` from the unsmoothed target avoids distorting off-target
    classes when ``label_smoothing > 0``. For best probability calibration in
    v4 training, prefer ``build_classification_loss(mode="cce_smooth")``.
    """

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
        """Compute focal loss using true-class ``p_t`` for the modulating factor."""
        y_true_hard = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # True-class probability p_t computed BEFORE label smoothing so off-target
        # smoothed labels do not receive an inflated (1 - y_pred)^gamma factor.
        p_t = tf.reduce_sum(y_true_hard * y_pred, axis=-1, keepdims=True)
        p_t = tf.clip_by_value(p_t, epsilon, 1.0 - epsilon)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        y_target = y_true_hard
        if self.label_smoothing > 0.0:
            num_classes = tf.cast(tf.shape(y_true_hard)[-1], tf.float32)
            y_target = y_true_hard * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        log_prob = tf.math.log(y_pred)
        cross_entropy = -y_target * log_prob

        if self.alpha is not None:
            alpha_tensor = tf.cast(self.alpha, tf.float32)
            cross_entropy = alpha_tensor * cross_entropy

        return tf.reduce_sum(modulating_factor * cross_entropy, axis=-1)

    def get_config(self) -> dict[str, Any]:
        """Return the serializable configuration dictionary."""
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
    """Construct the classification loss for v3/v4 training.

    Args:
        mode: ``"cce_smooth"`` (default for v4, preserves probability calibration
            when paired with capped class weights) or ``"focal"``.
        label_smoothing: Label smoothing factor in [0, 1). Default: 0.05.
        gamma: Focusing parameter used when ``mode == "focal"``.
        alpha: Optional per-class alpha weights when ``mode == "focal"``.
    """
    if mode == "cce_smooth":
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=float(label_smoothing))
    if mode == "focal":
        return CategoricalFocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)
    raise ValueError(f"Unsupported loss mode: {mode!r}. Expected 'cce_smooth' or 'focal'.")

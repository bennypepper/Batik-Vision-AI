"""Focal Categorical Crossentropy Loss for imbalanced classification."""

from typing import Any
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="batik_v3")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Categorical Focal Loss with label smoothing support.

    Formulation:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

    Attributes:
        gamma: Focusing parameter balancing easy and hard examples. Default: 2.0.
        alpha: Optional class weights tensor or list. Default: None.
        label_smoothing: Float in [0, 1]. Smooths categorical targets. Default: 0.1.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Any = None,
        label_smoothing: float = 0.1,
        reduction: str = "sum_over_batch_size",
        name: str = "categorical_focal_loss",
    ):
        super().__init__(reduction=reduction, name=name)
        self.gamma = float(gamma)
        self.alpha = alpha
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Computes focal loss between ground truth and predictions.

        Args:
            y_true: Ground truth one-hot encoded tensor.
            y_pred: Predicted probabilities tensor (softmax output).

        Returns:
            Computed focal loss tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.label_smoothing > 0.0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = tf.pow(1.0 - y_pred, self.gamma)
        log_prob = tf.math.log(y_pred)

        focal_cross_entropy = -y_true * modulating_factor * log_prob

        if self.alpha is not None:
            alpha_tensor = tf.cast(self.alpha, tf.float32)
            focal_cross_entropy = alpha_tensor * focal_cross_entropy

        return tf.reduce_sum(focal_cross_entropy, axis=-1)

    def get_config(self) -> dict[str, Any]:
        """Returns the serializable configuration dictionary."""
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
        })
        return config

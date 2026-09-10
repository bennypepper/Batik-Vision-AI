"""Cell 4: V3 Backbone model architecture, Categorical Focal Loss, and effective sample weighting."""

from typing import Sequence, Any
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="batik_v3")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Categorical Focal Loss with label smoothing support.

    Formulation:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
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
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.label_smoothing > 0.0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        modulating_factor = tf.pow(1.0 - y_pred, self.gamma)
        log_prob = tf.math.log(y_pred)

        focal_cross_entropy = -y_true * modulating_factor * log_prob

        if self.alpha is not None:
            alpha_tensor = tf.cast(self.alpha, tf.float32)
            focal_cross_entropy = alpha_tensor * focal_cross_entropy

        return tf.reduce_sum(focal_cross_entropy, axis=-1)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
        })
        return config


def compute_effective_sample_weights(
    class_counts: Sequence[int],
    beta: float = 0.999,
) -> np.ndarray:
    """Computes class weights based on Effective Number of Samples formulation.

    W_c = (1 - beta) / (1 - beta^{N_c})
    """
    counts = np.array(class_counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    effective_num = np.where(effective_num <= 0.0, 1e-8, effective_num)

    weights = (1.0 - beta) / effective_num
    normalized_weights = weights / np.sum(weights) * len(counts)
    return normalized_weights.astype(np.float32)


def create_v3_backbone_model(
    num_classes: int = 28,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str | None = "imagenet",
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Builds the v3 EfficientNetV2S classifier with single explicit Rescaling layer.

    Accepts raw [0, 255] float32 pixels. EfficientNetV2S has include_preprocessing=False.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image", dtype=tf.float32)

    # Explicit single rescaling: [0, 255] -> [-1, 1]
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

    # Classification head
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


print("Model architecture and Focal Loss modules ready.")

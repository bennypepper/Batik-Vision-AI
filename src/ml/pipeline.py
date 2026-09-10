"""Model architecture and data pipeline for Batik Motifs Detector v3."""

from typing import Sequence
import numpy as np
import tensorflow as tf


def create_v3_backbone_model(
    num_classes: int = 28,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str | None = None,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Creates the v3 EfficientNetV2S model with fixed preprocessing.

    The model accepts raw RGB float32 values in the range [0, 255] and handles
    rescaling internally via an explicit Rescaling layer. EfficientNetV2S is
    instantiated with include_preprocessing=False to prevent double normalization.

    Args:
        num_classes: Number of target classification categories. Default: 28.
        input_shape: Shape tuple of input tensor (height, width, channels).
        weights: Pretrained weights to initialize ('imagenet' or None).

    Returns:
        tuple of (base_model, full_model):
            - base_model: The underlying EfficientNetV2S feature extractor.
            - full_model: The complete functional model with rescaling and head.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image", dtype=tf.float32)

    # Single explicit rescaling layer: [0, 255] -> [-1, 1]
    rescaled = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="input_rescaling",
    )(inputs)

    # Base feature extractor without internal preprocessing
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


def compute_effective_sample_weights(
    class_counts: Sequence[int],
    beta: float = 0.999,
) -> np.ndarray:
    """Computes class weights based on the Effective Number of Samples formulation.

    Reference: Cui et al., Class-Balanced Loss Based on Effective Number of Samples (CVPR 2019).
    W_c = (1 - beta) / (1 - beta^{N_c})

    Args:
        class_counts: List or sequence of sample counts per class.
        beta: Hyperparameter controlling the scale of effective samples (default: 0.999).

    Returns:
        np.ndarray of float32 weights normalized such that sum(weights) == len(class_counts).
    """
    counts = np.array(class_counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    effective_num = np.where(effective_num <= 0.0, 1e-8, effective_num)

    weights = (1.0 - beta) / effective_num
    normalized_weights = weights / np.sum(weights) * len(counts)
    return normalized_weights.astype(np.float32)

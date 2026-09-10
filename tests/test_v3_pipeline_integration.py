"""Integration tests for model compilation, forward pass, and training step."""

import numpy as np
import pytest
import tensorflow as tf
from src.ml.pipeline import create_v3_backbone_model
from src.ml.losses import CategoricalFocalLoss


def test_v3_compilation_and_step():
    """Verifies that the v3 model compiles with Focal Loss and completes a training step."""
    base, model = create_v3_backbone_model(num_classes=5)
    model.compile(
        optimizer="adam",
        loss=CategoricalFocalLoss(gamma=2.0),
        metrics=["accuracy"],
    )
    assert model.loss is not None

    # Verify a single train step executes without shape or type errors
    x = np.random.uniform(0.0, 255.0, size=(2, 224, 224, 3)).astype(np.float32)
    y = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    metrics = model.train_on_batch(x, y)
    assert metrics is not None

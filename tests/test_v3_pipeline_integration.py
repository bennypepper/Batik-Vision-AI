"""Integration tests for model compilation, forward pass, and training step."""

import numpy as np

from src.ml.losses import CategoricalFocalLoss
from src.ml.pipeline import create_v3_backbone_model


def test_v3_compilation_and_step():
    """Verify that the v3 model compiles with Focal Loss and completes a training step."""
    _, model = create_v3_backbone_model(num_classes=5)
    model.compile(
        optimizer="adam",
        loss=CategoricalFocalLoss(gamma=2.0),
        metrics=["accuracy"],
    )
    assert model.loss is not None

    x = np.random.uniform(0.0, 255.0, size=(2, 224, 224, 3)).astype(np.float32)
    y = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    metrics = model.train_on_batch(x, y)
    assert metrics is not None

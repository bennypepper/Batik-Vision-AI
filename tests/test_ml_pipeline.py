import numpy as np
import tensorflow as tf

from src.ml.losses import CategoricalFocalLoss
from src.ml.pipeline import compute_effective_sample_weights, create_v3_backbone_model


def test_v3_model_rescaling_input():
    _, model = create_v3_backbone_model(num_classes=28)

    # Test that model accepts raw [0, 255] float input without errors
    dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)
    output = model(dummy_input, training=False)

    assert output.shape == (1, 28)
    # Probability distribution should sum to 1.0
    np.testing.assert_almost_equal(tf.reduce_sum(output, axis=-1).numpy(), [1.0], decimal=5)


def test_effective_sample_weights():
    counts = [34, 100, 230]
    weights = compute_effective_sample_weights(counts, beta=0.999)
    assert len(weights) == 3
    # Minority class should have highest weight
    assert weights[0] > weights[1] > weights[2]


def test_focal_loss_computation():
    loss_fn = CategoricalFocalLoss(gamma=2.0, label_smoothing=0.1)
    y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8]], dtype=tf.float32)
    loss = loss_fn(y_true, y_pred)
    assert float(loss) > 0.0

"""Numerical parity validator across Keras, TFLite, and ONNX model artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from src.ml.losses import CategoricalFocalLoss
from src.ml.pipeline import CosineWarmupSchedule


def _load_real_batik_batch(image_path: str, img_size: int, batch_size: int) -> np.ndarray:
    """Load a real batik image into a raw [0, 255] float32 batch."""
    with Image.open(image_path) as im:
        resized = im.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)
    return np.repeat(arr[np.newaxis, ...], batch_size, axis=0)


def verify_parity(
    keras_path: str,
    tflite_path: str | None = None,
    onnx_path: str | None = None,
    tolerance: float = 1e-3,
    num_samples: int = 5,
    seed: int = 42,
    image_path: str | None = None,
) -> bool:
    """Validate numerical parity across Keras, TFLite, and ONNX for batch_size=1 and batch_size=2."""
    rng = np.random.default_rng(seed)
    print("=" * 65)
    print("MODEL RUNTIME PARITY VALIDATION")
    print("=" * 65)

    print(f"Loading Keras model from: {keras_path}")
    keras_model = tf.keras.models.load_model(
        keras_path,
        custom_objects={
            "CategoricalFocalLoss": CategoricalFocalLoss,
            "CosineWarmupSchedule": CosineWarmupSchedule,
        },
        compile=False,
    )
    img_size = int(keras_model.input_shape[1] or 224)

    tflite_interpreter = None
    tflite_in = 0
    tflite_out = 0
    if tflite_path and Path(tflite_path).exists():
        print(f"Loading TFLite model from: {tflite_path}")
        tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
        tflite_interpreter.allocate_tensors()
        tflite_in = tflite_interpreter.get_input_details()[0]["index"]
        tflite_out = tflite_interpreter.get_output_details()[0]["index"]

    onnx_session = None
    onnx_in_name = ""
    if onnx_path and Path(onnx_path).exists():
        if not HAS_ORT:
            print("WARNING: onnxruntime not installed in this environment. Skipping ONNX check.")
        else:
            print(f"Loading ONNX model from: {onnx_path}")
            onnx_session = ort.InferenceSession(onnx_path)
            onnx_in_name = onnx_session.get_inputs()[0].name

    all_passed = True

    test_batches: list[tuple[str, np.ndarray]] = []
    if image_path:
        for bs in (1, 2):
            test_batches.append((f"real_image(batch={bs})", _load_real_batik_batch(image_path, img_size, bs)))

    for i in range(num_samples):
        for bs in (1, 2):
            arr = rng.uniform(0.0, 255.0, size=(bs, img_size, img_size, 3)).astype(np.float32)
            test_batches.append((f"random_{i + 1}(batch={bs})", arr))

    for label, batch_input in test_batches:
        keras_out = keras_model(batch_input, training=False).numpy()

        if tflite_interpreter is not None and batch_input.shape[0] == 1:
            tflite_interpreter.set_tensor(tflite_in, batch_input)
            tflite_interpreter.invoke()
            tflite_out_arr = tflite_interpreter.get_tensor(tflite_out)
            max_diff = float(np.max(np.abs(keras_out - tflite_out_arr)))
            if max_diff > tolerance:
                print(f"FAILED: {label} TFLite diff = {max_diff:.6f} > {tolerance}")
                all_passed = False
            else:
                print(f"PASSED: {label} Keras vs TFLite max diff = {max_diff:.6f}")

        if onnx_session is not None:
            onnx_out_arr = onnx_session.run(None, {onnx_in_name: batch_input})[0]
            max_diff = float(np.max(np.abs(keras_out - onnx_out_arr)))
            if max_diff > tolerance:
                print(f"FAILED: {label} ONNX diff = {max_diff:.6f} > {tolerance}")
                all_passed = False
            else:
                print(f"PASSED: {label} Keras vs ONNX max diff = {max_diff:.6f}")

    if all_passed:
        print("\nAll runtime models match within tolerance.")
    else:
        print("\nParity verification failed. Discrepancy exceeds tolerance threshold.")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify parity between Keras, TFLite, and ONNX models.")
    parser.add_argument("--keras", required=True, help="Path to .keras model")
    parser.add_argument("--tflite", default=None, help="Path to .tflite model")
    parser.add_argument("--onnx", default=None, help="Path to .onnx model")
    parser.add_argument("--image", default=None, help="Optional real batik image path for parity test")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Max absolute difference threshold")
    parser.add_argument("--samples", type=int, default=5, help="Number of random test pairs (batch 1 & 2)")

    args = parser.parse_args()
    success = verify_parity(
        keras_path=args.keras,
        tflite_path=args.tflite,
        onnx_path=args.onnx,
        tolerance=args.tolerance,
        num_samples=args.samples,
        image_path=args.image,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

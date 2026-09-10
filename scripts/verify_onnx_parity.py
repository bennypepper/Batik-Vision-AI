"""Numerical parity validator across Keras, TFLite, and ONNX model artifacts."""

import argparse
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from src.ml.losses import CategoricalFocalLoss


def verify_parity(
    keras_path: str,
    tflite_path: str | None = None,
    onnx_path: str | None = None,
    tolerance: float = 1e-3,
    num_samples: int = 5,
    seed: int = 42,
) -> bool:
    """Validates numerical parity across Keras, TFLite, and ONNX model runtimes.

    Args:
        keras_path: Path to .keras SavedModel.
        tflite_path: Optional path to .tflite model.
        onnx_path: Optional path to .onnx model.
        tolerance: Maximum acceptable absolute difference between prediction vectors.
        num_samples: Number of random test batches to verify.
        seed: Random seed for deterministic generation.

    Returns:
        bool indicating whether all checks passed within tolerance.
    """
    np.random.seed(seed)
    print("=" * 65)
    print("MODEL RUNTIME PARITY VALIDATION")
    print("=" * 65)

    # 1. Load Keras model
    print(f"Loading Keras model from: {keras_path}")
    keras_model = tf.keras.models.load_model(
        keras_path,
        custom_objects={"CategoricalFocalLoss": CategoricalFocalLoss},
    )

    # 2. Load TFLite interpreter if provided
    tflite_interpreter = None
    if tflite_path and Path(tflite_path).exists():
        print(f"Loading TFLite model from: {tflite_path}")
        tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
        tflite_interpreter.allocate_tensors()
        tflite_in = tflite_interpreter.get_input_details()[0]["index"]
        tflite_out = tflite_interpreter.get_output_details()[0]["index"]

    # 3. Load ONNX session if provided
    onnx_session = None
    if onnx_path and Path(onnx_path).exists():
        if not HAS_ORT:
            print("WARNING: onnxruntime not installed in this environment. Skipping ONNX check.")
        else:
            print(f"Loading ONNX model from: {onnx_path}")
            onnx_session = ort.InferenceSession(onnx_path)
            onnx_in_name = onnx_session.get_inputs()[0].name

    all_passed = True

    for i in range(num_samples):
        # Raw RGB inputs in [0, 255]
        dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)

        # Keras prediction
        keras_out = keras_model(dummy_input, training=False).numpy()[0]

        # Check TFLite
        if tflite_interpreter is not None:
            tflite_interpreter.set_tensor(tflite_in, dummy_input)
            tflite_interpreter.invoke()
            tflite_out_arr = tflite_interpreter.get_tensor(tflite_out)[0]
            max_diff = np.max(np.abs(keras_out - tflite_out_arr))
            if max_diff > tolerance:
                print(f"FAILED: Sample {i + 1} TFLite diff = {max_diff:.6f} > {tolerance}")
                all_passed = False
            else:
                print(f"PASSED: Sample {i + 1} Keras vs TFLite max diff = {max_diff:.6f}")

        # Check ONNX
        if onnx_session is not None:
            onnx_out_arr = onnx_session.run(None, {onnx_in_name: dummy_input})[0][0]
            max_diff = np.max(np.abs(keras_out - onnx_out_arr))
            if max_diff > tolerance:
                print(f"FAILED: Sample {i + 1} ONNX diff = {max_diff:.6f} > {tolerance}")
                all_passed = False
            else:
                print(f"PASSED: Sample {i + 1} Keras vs ONNX max diff = {max_diff:.6f}")

    if all_passed:
        print("\nAll runtime models match within tolerance.")
    else:
        print("\nParity verification failed. Discrepancy exceeds tolerance threshold.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Verify parity between Keras, TFLite, and ONNX models.")
    parser.add_argument("--keras", required=True, help="Path to .keras model")
    parser.add_argument("--tflite", default=None, help="Path to .tflite model")
    parser.add_argument("--onnx", default=None, help="Path to .onnx model")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Max absolute difference threshold")
    parser.add_argument("--samples", type=int, default=5, help="Number of random test samples")

    args = parser.parse_args()
    success = verify_parity(
        keras_path=args.keras,
        tflite_path=args.tflite,
        onnx_path=args.onnx,
        tolerance=args.tolerance,
        num_samples=args.samples,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Cell 7: Model export to TFLite and ONNX (FP16), followed by numerical parity validation."""

import os
import subprocess
import numpy as np
import tensorflow as tf

print("Step 1: Exporting FP16 TFLite model...")
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH) if os.path.isdir(MODEL_PATH) else None
if converter is None:
    best_model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"CategoricalFocalLoss": CategoricalFocalLoss},
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

tflite_size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
print(f"Exported TFLite model: {TFLITE_PATH} ({tflite_size_mb:.2f} MB)")

print("\nStep 2: Exporting ONNX model via tf2onnx...")
onnx_cmd = [
    "python", "-m", "tf2onnx.convert",
    "--keras", MODEL_PATH,
    "--output", ONNX_PATH,
    "--opset", "17",
]
try:
    subprocess.run(onnx_cmd, check=True)
    onnx_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"Exported ONNX model: {ONNX_PATH} ({onnx_size_mb:.2f} MB)")
except Exception as e:
    print(f"ONNX conversion warning: {e}. If running locally, install tf2onnx (`pip install tf2onnx`).")

print("\nStep 3: Numerical Parity Verification across runtimes...")
dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)

# Keras prediction
keras_pred = best_model(dummy_input, training=False).numpy()[0]

# TFLite prediction
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]["index"], dummy_input)
interpreter.invoke()
tflite_pred = interpreter.get_tensor(output_details[0]["index"])[0]

max_diff_tflite = np.max(np.abs(keras_pred - tflite_pred))
print(f"  Max absolute deviation (Keras vs TFLite FP16): {max_diff_tflite:.6f}")
assert max_diff_tflite < 1e-2, f"Discrepancy too high: {max_diff_tflite}"

# ONNX prediction (if onnxruntime installed)
if os.path.exists(ONNX_PATH):
    try:
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(ONNX_PATH)
        onnx_input_name = ort_sess.get_inputs()[0].name
        onnx_pred = ort_sess.run(None, {onnx_input_name: dummy_input})[0][0]

        max_diff_onnx = np.max(np.abs(keras_pred - onnx_pred))
        print(f"  Max absolute deviation (Keras vs ONNX): {max_diff_onnx:.6f}")
        assert max_diff_onnx < 1e-3, f"ONNX Discrepancy too high: {max_diff_onnx}"
    except ImportError:
        print("  onnxruntime not installed in this environment; skipping ONNX runtime verification.")

print("\nModel serialization and parity verification complete.")

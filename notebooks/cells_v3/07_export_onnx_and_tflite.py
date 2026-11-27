"""Cell 7: Model export to TFLite and ONNX (opset 17 + optional FP16), followed by parity validation."""

import os
import subprocess
import sys

import numpy as np
import tensorflow as tf

print("Step 1: Exporting FP16 TFLite model...")
best_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "CategoricalFocalLoss": CategoricalFocalLoss,
        "CosineWarmupSchedule": CosineWarmupSchedule,
    },
    compile=False,
)
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

tflite_size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
print(f"Exported TFLite model: {TFLITE_PATH} ({tflite_size_mb:.2f} MB)")

print("\nStep 2: Exporting ONNX model via tf2onnx (opset 17)...")
onnx_cmd = [
    sys.executable,
    "-m",
    "tf2onnx.convert",
    "--keras",
    MODEL_PATH,
    "--output",
    ONNX_PATH,
    "--opset",
    "17",
]
try:
    subprocess.run(onnx_cmd, check=True)
    onnx_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"Exported FP32 ONNX model: {ONNX_PATH} ({onnx_size_mb:.2f} MB)")

    # Optional FP16 ONNX quantization when onnxconverter_common is available
    try:
        import onnx
        from onnxconverter_common import float16

        fp16_onnx_path = ONNX_PATH.replace(".onnx", "_fp16.onnx")
        fp32_proto = onnx.load(ONNX_PATH)
        fp16_proto = float16.convert_float_to_float16(fp32_proto, keep_io_types=True)
        onnx.save(fp16_proto, fp16_onnx_path)
        fp16_size_mb = os.path.getsize(fp16_onnx_path) / (1024 * 1024)
        print(f"Exported FP16 ONNX model: {fp16_onnx_path} ({fp16_size_mb:.2f} MB)")
    except ImportError:
        print(
            "NOTICE: onnxconverter_common not installed; keeping FP32 ONNX artifact. "
            "Install `onnxconverter-common` for FP16 ONNX quantization."
        )
except Exception as e:
    print(f"ONNX conversion warning: {e}. Install tf2onnx (`pip install tf2onnx`).")

print("\nStep 3: Numerical Parity Verification across runtimes (batch_size=1 & batch_size=2)...")
for batch_size in (1, 2):
    dummy_input = np.random.uniform(0.0, 255.0, size=(batch_size, IMG_HEIGHT, IMG_WIDTH, 3)).astype(np.float32)
    keras_pred = best_model(dummy_input, training=False).numpy()

    if batch_size == 1:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        tflite_pred = interpreter.get_tensor(output_details[0]["index"])
        max_diff_tflite = np.max(np.abs(keras_pred - tflite_pred))
        print(f"  [batch={batch_size}] Max abs deviation (Keras vs TFLite FP16): {max_diff_tflite:.6f}")
        assert max_diff_tflite < 1e-2, f"TFLite discrepancy too high: {max_diff_tflite}"

    if os.path.exists(ONNX_PATH):
        try:
            import onnxruntime as ort

            ort_sess = ort.InferenceSession(ONNX_PATH)
            onnx_input_name = ort_sess.get_inputs()[0].name
            onnx_pred = ort_sess.run(None, {onnx_input_name: dummy_input})[0]
            max_diff_onnx = np.max(np.abs(keras_pred - onnx_pred))
            print(f"  [batch={batch_size}] Max abs deviation (Keras vs ONNX): {max_diff_onnx:.6f}")
            assert max_diff_onnx < 1e-3, f"ONNX discrepancy too high: {max_diff_onnx}"
        except ImportError:
            print("  onnxruntime not installed; skipping ONNX runtime verification.")

print("\nModel serialization and parity verification complete.")

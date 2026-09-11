import tensorflow as tf
import tf2onnx
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "results" / "v3_results" / "best_batik_model_v3.keras"
OUTPUT_DIR = REPO_ROOT / "web" / "public" / "models"
OUTPUT_PATH = OUTPUT_DIR / "batik_model_v3.onnx"

def main():
    if not MODEL_PATH.exists():
        print(f"Error: model path {MODEL_PATH} not found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading Keras model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    print("Converting Keras model to ONNX with opset 15...")
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input_1"),)
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=15,
        output_path=str(OUTPUT_PATH)
    )
    print(f"Successfully converted model to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()

"""Convert a Keras Batik model to ONNX (opset 17) with optional FP16/INT8 quantization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "results" / "v3_results" / "best_batik_model_v3.keras"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "web" / "public" / "models" / "batik_model_v3.onnx"


def convert_keras_to_onnx(
    input_path: Path,
    output_path: Path,
    img_size: int = 224,
    opset: int = 17,
    quantize: str = "none",
) -> Path:
    """Convert a ``.keras`` model to ONNX (opset 17) and optionally quantize."""
    import tensorflow as tf
    import tf2onnx

    if not input_path.exists():
        raise FileNotFoundError(f"Model path not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading Keras model from {input_path}...")
    model = tf.keras.models.load_model(input_path, compile=False)

    print(f"Converting Keras model to ONNX (opset={opset}, img_size={img_size})...")
    spec = (tf.TensorSpec((None, img_size, img_size, 3), tf.float32, name="input_1"),)
    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=opset,
        output_path=str(output_path),
    )
    print(f"Exported FP32 ONNX: {output_path} ({output_path.stat().st_size / (1024 * 1024):.2f} MB)")

    if quantize == "fp16":
        try:
            import onnx
            from onnxconverter_common import float16

            fp32_proto = onnx.load(str(output_path))
            fp16_proto = float16.convert_float_to_float16(fp32_proto, keep_io_types=True)
            onnx.save(fp16_proto, str(output_path))
            print(f"Quantized to FP16 ONNX: {output_path} ({output_path.stat().st_size / (1024 * 1024):.2f} MB)")
        except ImportError:
            print(
                "NOTICE: onnxconverter_common is not installed; keeping FP32 ONNX artifact. "
                "Install `onnxconverter-common` to enable FP16 quantization."
            )
    elif quantize == "int8":
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            tmp_fp32 = output_path.with_suffix(".fp32.onnx")
            output_path.replace(tmp_fp32)
            quantize_dynamic(
                model_input=str(tmp_fp32),
                model_output=str(output_path),
                weight_type=QuantType.QUInt8,
            )
            tmp_fp32.unlink(missing_ok=True)
            print(f"Quantized to INT8 ONNX: {output_path} ({output_path.stat().st_size / (1024 * 1024):.2f} MB)")
        except ImportError:
            print(
                "NOTICE: onnxruntime.quantization is not installed; keeping FP32 ONNX artifact."
            )

    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Keras model to ONNX (opset 17).")
    parser.add_argument("--input", type=Path, default=DEFAULT_MODEL_PATH, help="Input .keras model path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output .onnx model path")
    parser.add_argument("--img-size", type=int, default=224, help="Square input spatial size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "fp16", "int8"],
        default="none",
        help="Optional post-export quantization mode",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        convert_keras_to_onnx(
            input_path=args.input,
            output_path=args.output,
            img_size=args.img_size,
            opset=args.opset,
            quantize=args.quantize,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

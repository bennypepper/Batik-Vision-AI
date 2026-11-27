"""Assembles modular cell scripts into notebooks/batik_motifs_v3_train.ipynb."""

import json
from pathlib import Path


def create_notebook():
    cells_dir = Path("notebooks/cells_v3")
    output_notebook_path = Path("notebooks/batik_motifs_v3_train.ipynb")

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9",
            },
            "kaggle": {
                "accelerator": "gpu",
                "isGpuEnabled": True,
                "isInternetEnabled": True,
            },
        },
        "cells": [],
    }

    # Intro markdown cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Indonesian Batik Motifs Classification v3\n",
            "\n",
            "This notebook implements the v3 architecture and training pipeline for the Indonesian Batik Motifs Detector.\n",
            "\n",
            "### Major Architectural Fixes in v3:\n",
            "1. **Double-Preprocessing Bug Resolved**: Fixed the dynamic range collapse where v2 normalized images twice ($[-1.008, -0.992]$). The model now has a single internal `Rescaling(scale=1/127.5, offset=-1.0)` layer and natively receives raw $[0, 255]$ float32 pixels.\n",
            "2. **Perceptual Hash Deduplication**: Uses 64-bit dHash and Hamming distance to eliminate train/val/test data leakage from redundant duplicate images.\n",
            "3. **Calibrated Loss & Capped Class Weighting**: Defaults to label-smoothed Categorical Crossentropy (`label_smoothing=0.05`) with capped Effective Number weights (`max_weight=3.0`), plus fixed true-class `p_t` Focal Loss ($\\gamma = 2.0$).\n",
            "4. **StratifiedGroupKFold 5-Fold Cross-Validation**: Two-phase fine-tuning (frozen backbone head training followed by top 120 layers unfreezing with cosine warmup and frozen BatchNormalization).\n",
            "5. **Zero-Backend Export**: Automatic serialization to FP16 TFLite and ONNX for client-side in-browser WebAssembly execution.",
        ],
    })

    cell_configs = [
        ("01_config_and_env.py", "## 1. Environment Configuration and Hyperparameters"),
        ("02_dataset_discovery_and_audit.py", "## 2. Dataset Discovery, Deduplication, and Class Profiling"),
        ("03_augmentation_and_tfdata.py", "## 3. Stratified Holdout, Tier-Based Augmentations, and Data Pipelines"),
        ("04_model_architecture_and_loss.py", "## 4. Model Architecture, Explicit Rescaling, and Focal Loss Formulation"),
        ("05_kfold_training_loop.py", "## 5. Stratified 5-Fold Training Loop (Phase 1 & Phase 2 Fine-Tuning)"),
        ("06_evaluation_and_metrics.py", "## 6. Holdout Evaluation, Test-Time Augmentation (TTA), and Classification Report"),
        ("07_export_onnx_and_tflite.py", "## 7. Model Serialization (TFLite & ONNX FP16) and Parity Verification"),
    ]

    for filename, heading in cell_configs:
        file_path = cells_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing cell file: {file_path}")

        code_lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)

        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [heading + "\n"],
        })

        notebook["cells"].append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": code_lines,
        })

    with open(output_notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Successfully assembled {output_notebook_path} with {len(notebook['cells'])} cells.")


if __name__ == "__main__":
    create_notebook()

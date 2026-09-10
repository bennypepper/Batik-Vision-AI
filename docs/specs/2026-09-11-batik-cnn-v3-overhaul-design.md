# CNN Batik Motifs Detector v3: System Architecture & Overhaul Specification

**Author:** Benedict Michael Pepper & Engineering Pair  
**Date:** 2026-09-11  
**Target Repository:** `CNN-Batik-Motifs-Detector`  
**Status:** Approved for Implementation (Phase 1)

---

## 1. Executive Summary

This specification defines the comprehensive overhaul of the **CNN Batik Motifs Detector** from an experimental academic prototype (v2.0) to a production-grade dual-tier machine learning system with a modern web interface (v3.0).

### Key Architectural Shifts
1. **Core Pipeline Defect Fix**: Eradicate the double-preprocessing defect where input pixels were compressed to less than 2% of their dynamic range during training.
2. **Two-Stage Inference Pipeline**: Integrate a lightweight client-side YOLO detector (YOLOv8n/11n in ONNX) to isolate the fabric region (eliminating human faces, limbs, and room backgrounds) prior to classification by the fine-tuned EfficientNetV2S model.
3. **Dataset Hygiene & Imbalance Regularization**: Execute perceptual hash (pHash) deduplication across splits and incorporate Focal Categorical Crossentropy ($\gamma = 2.0$) with effective sample class weighting to resolve poor performance on minority and ambiguous classes (`Priangan Merak Ngibing`, `Sogan`, `Lasem`).
4. **Next.js Web Application (`web/`)**: Replace the monolithic Streamlit prototype with an interactive Next.js application running in-browser client-side inference via `onnxruntime-web` (WebAssembly/WebGL).

---

## 2. Directory Layout (Integrated Monorepo)

```
CNN-Batik-Motifs-Detector/
├── models/
│   ├── batik_model_v2.tflite          # Existing v2 baseline (FP16)
│   ├── batik_model_v3.onnx            # v3 production model for web runtime
│   ├── batik_model_v3.tflite          # v3 mobile/Python artifact
│   └── labels.json                    # Canonical 28 classes + regional metadata
├── notebooks/
│   ├── batik_motifs_v3_train.ipynb    # Clean Kaggle/Colab training notebook
│   └── cells/                         # Exported modular Python execution cells
├── docs/
│   ├── ARCHITECTURE_v2.md
│   ├── KNOWN_ISSUES.md
│   └── specs/                         # Technical specifications
│       └── 2026-09-11-batik-cnn-v3-overhaul-design.md
├── scripts/
│   ├── audit_dataset_leakage.py       # pHash deduplication & leakage detector
│   ├── convert_to_onnx.py             # Keras to ONNX conversion and quantization
│   └── verify_onnx_parity.py          # Parity validation across Keras, TFLite, ONNX
├── web/                               # Next.js 14+ App Router application
│   ├── public/
│   │   ├── models/
│   │   │   ├── batik_model_v3.onnx    # In-browser classification model
│   │   │   └── yolo_batik_crop.onnx   # Lightweight fabric region detector
│   │   └── samples/                   # High-res benchmark motif samples
│   ├── src/
│   │   ├── app/                       # App Router routes (/scanner, /atlas, /diagnostics)
│   │   ├── components/                # React UI components (Scanner, BBoxCanvas, MotifCard)
│   │   ├── lib/
│   │   │   ├── onnx-session.ts        # onnxruntime-web session management
│   │   │   ├── image-processing.ts    # Canvas resize, crop, and tensor packing
│   │   └── batik-data.ts              # 28 motif metadata, origins, and philosophies
│   │   └── types/
│   ├── package.json
│   ├── tsconfig.json
│   └── tailwind.config.ts
├── requirements.txt                   # ML and conversion dependencies
└── README.md
```

---

## 3. Machine Learning Pipeline Specification (v3.0)

### 3.1 Resolving the Double-Preprocessing Bug
In v2, `preprocess_input(img)` in the data pipeline scaled images to $[-1.0, 1.0]$, while `EfficientNetV2S` internally applied a default `Rescaling` layer, collapsing inputs to $[-1.008, -0.992]$.

**V3 Solution**:
* The `EfficientNetV2S` backbone is instantiated with `include_preprocessing=False`.
* A single explicit `tf.keras.layers.Rescaling(scale=1.0/127.5, offset=-1.0)` layer is placed at the top of the Keras functional model.
* The model natively accepts raw RGB $[0, 255]$ values. Both training (`tf.data`) and client-side inference (`HTML5 Canvas ImageData`) feed raw $[0, 255]$ pixels, ensuring strict mathematical parity.

### 3.2 Loss Formulation: Focal Categorical Crossentropy
To address class imbalance (from 34 images in `Priangan_Merak_Ngibing` to 230 in `Kalimantan_Dayak`), standard categorical cross-entropy is replaced by Focal Loss:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

* Focusing parameter $\gamma = 2.0$.
* Label smoothing parameter $= 0.1$.
* Class weights computed via the Effective Number of Samples formulation:

$$W_c = \frac{1 - \beta}{1 - \beta^{N_c}}, \quad \beta = 0.999$$

### 3.3 Training Phases & Optimization
* **Phase 1 (Backbone Frozen)**:
  * Trains the classification head: `GAP -> BatchNormalization -> Dropout(0.5) -> Dense(256) -> Dropout(0.3) -> Dense(28) -> Softmax(float32)`.
  * Optimizer: Adam ($\text{lr} = 10^{-3}$), patience $= 7$ on validation loss.
* **Phase 2 (Selective Fine-Tuning)**:
  * Unfreeze top 40 backbone layers; keep all `BatchNormalization` layers frozen.
  * Optimizer: Cosine decay learning rate with 2-epoch linear warmup from $10^{-6}$ to $10^{-5}$, decaying to $10^{-7}$.
  * Early stopping patience $= 10$.

### 3.4 Model Serialization and Verification Targets
* `batik_model_v3.keras`: Full Keras SavedModel.
* `batik_model_v3.onnx`: Exported via `tf2onnx` with OPSET 17, quantized to FP16.
* `batik_model_v3.tflite`: FP16 quantized TFLite artifact.
* `scripts/verify_onnx_parity.py`: Validates that Keras, TFLite, and ONNX outputs deviate by less than $10^{-4}$ on sample inputs.

---

## 4. Two-Stage Detection Architecture

1. **Stage 1 (YOLO Detector)**:
   * A compact YOLO model (`yolo_batik_crop.onnx`, $\approx 6$ MB) runs client-side.
   * Identifies the primary fabric / clothing bounding box from unconstrained photos.
   * Displays an interactive bounding box on the camera/upload canvas.
2. **Interactive ROI Cropping**:
   * Automatic crop based on maximum confidence bounding box.
   * User can optionally drag/resize the crop box to target specific patterns.
3. **Stage 2 (EfficientNet Classifier)**:
   * The cropped texture is resized to $224 \times 224$ and passed into `batik_model_v3.onnx`.
   * Returns Top-1 prediction, Top-5 candidate probabilities, and regional metadata.

---

## 5. Dataset Hygiene Plan (Phase 1 Target)

* **Dataset Source**: `D:\Personal Projects\Computer Vision-ML Projects\Batik Motifs Detector v2\Batik_dataset\dataset`
* **Audit Script (`scripts/audit_dataset_leakage.py`)**:
  * Calculates difference hashes (dHash) and perceptual hashes (pHash) for all images.
  * Detects duplicate or near-duplicate images ($hamming\_distance \le 3$) across class folders.
  * Outputs summary report and CSV of flagged pairs.
* **Class Mapping & Validation**:
  * Verifies the 28 active classes and generates `models/labels.json` with canonical IDs, Indonesian names, English names, and regional provenance.

---

## 6. Implementation Phases

| Phase | Description | Deliverables |
| :--- | :--- | :--- |
| **Phase 1** | Dataset Audit, Hygiene & Notebook Refactoring | `scripts/audit_dataset_leakage.py`, `models/labels.json`, updated `notebooks/batik_motifs_v3_train.ipynb` |
| **Phase 2** | Retraining, Model Conversion & Parity Tests | `batik_model_v3.keras`, `batik_model_v3.onnx`, `batik_model_v3.tflite`, `scripts/verify_onnx_parity.py` |
| **Phase 3** | Next.js Frontend Scaffolding & Web Worker ONNX Engine | `web/` initialized, `onnxruntime-web` pipeline, canvas processing library |
| **Phase 4** | Two-Stage UI, Cultural Atlas & Philosophy Drawer | Scanner component, interactive BBox cropper, 28-motif Atlas, bilingual support |
| **Phase 5** | Verification, Benchmarking & Deployment | Full integration tests, repository audit, Vercel deployment |

---

## 7. Spec Self-Review
* **Placeholders**: None. All hyperparameter values, formulas, and paths are explicitly specified.
* **Consistency**: Model input formats ($[0, 255]$ raw) match across training, ONNX export, and Next.js canvas processing.
* **Scope**: Phase 1 is strictly scoped to dataset auditing, label consolidation, and training pipeline modernization.

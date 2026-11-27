# Batik Vision AI: Two-Stage Indonesian Batik Motifs Recognition Engine

Batik Vision AI is a computer vision platform designed to preserve, document, and classify 28 traditional Indonesian batik motifs. The system combines client-side adaptive fabric localization (with Stage-1 ONNX garment segmentation support) and aspect-preserving multi-patch texture classification via EfficientNetV2S, executed entirely inside a browser Web Worker using ONNX Runtime Web (WebGPU / WebGL / self-hosted WebAssembly).

## System Architecture

```
                       [ Input Image: Textile or Garment ]
                                        │
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │   Stage 1: Adaptive Fabric Localization  │
                   │  - 256px resolution-normalized 6x6 grid  │
                   │    Laplacian texture saliency detector   │
                   │  - Full Swatch, Center & Adaptive ROIs   │
                   │  - Touch-friendly manual ROI cropper     │
                   └────────────────────┬─────────────────────┘
                                        │
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │ Stage 2: Multi-Patch Motif Classifier    │
                   │  - 3 aspect-preserving 1:1 square tiles  │
                   │  - Backbone: Fine-Tuned EfficientNetV2S  │
                   │  - Internal Rescaling: [0,255] -> [-1,1] │
                   │  - Entropy-weighted voting + OOD check   │
                   └────────────────────┬─────────────────────┘
                                        │
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │   Zero-Backend Client-Side Web Worker    │
                   │  - ONNX Runtime WebGPU / WebGL / WASM    │
                   │  - Self-hosted /ort/ WASM (no CDN dep)   │
                   │  - Cache API persistence & Cultural Atlas│
                   └──────────────────────────────────────────┘
```

## Architectural Improvements: v2 → v3 → v4 Pipeline

### 1. Elimination of Double Preprocessing
In v2, input images underwent double normalization (`[0, 255]` → `[-1, 1]` twice, collapsing dynamic range to `[-1.008, -0.992]`). In v3/v4, `EfficientNetV2S` is instantiated with `include_preprocessing=False` and a single explicit internal `Rescaling(scale=1.0/127.5, offset=-1.0)` layer accepting raw `[0, 255]` `float32` tensors.

### 2. Leak-Free Deduplication & Visual Cluster Splitting
64-bit difference hashing (dHash) with cross-class conflict quarantine (`scripts/audit_dataset_leakage.py`) and optional DINOv2 cosine-similarity clustering (`scripts/train_v4_colab.py --use-dinov2-clustering`) group near-duplicate and multi-angle garment shots into `GroupStratifiedKFold` clusters so identical fabrics never leak across train/validation/test splits.

### 3. Domain-Safe Augmentations & Calibrated Class Weighting
`src/ml/augmentations.py` keeps raw high-resolution `uint8` RGB frames through Albumentations (cropping before resizing) and protects batik-specific semantics: horizontal flips are disabled for orientation-sensitive motifs (`Solo_Parang`, `Yogyakarta_Parang`, `Sulawesi_Selatan_Lontara`), hue jitter is bounded (`<= 5`) to protect dye-defined classes (`Sogan`, `Lasem`), and `CategoricalCrossentropy(label_smoothing=0.05)` is paired with capped Effective Number weights (`max <= 3.0`) + post-hoc Temperature Scaling (`src/ml/calibration.py`) and Out-Of-Distribution (OOD) rejection.

### 4. Resolution-Normalized Stage-1 Localization & Multi-Patch Voting
Stage 1 (`web/src/lib/yolo-detector.ts`) normalizes images to `256px` and computes a `6×6` grid of Laplacian texture energy to locate both flat swatches and off-center garments. Stage 2 (`web/src/lib/classifier.ts`) extracts 3 aspect-preserving 1:1 square tiles from the ROI and merges predictions via inverse-entropy weighted voting — eliminating 16:9→224×224 geometric squashing.

### 5. Dedicated Web Worker + Self-Hosted ONNX Runtime Web
Inference runs off the main thread in `web/src/lib/inference.worker.ts` using `executionProviders: ["webgpu", "webgl", "wasm"]`, self-hosted `/ort/` WASM assets, explicit tensor disposal, Cache API persistence, download progress reporting, and zero silent fake fallbacks.

## Supported Batik Motifs (28 Classes)

| # | Motif Identifier | Indonesian Name | English Name | Region | Island |
|---|---|---|---|---|---|
| 1 | Bali_Barong | Bali Barong | Bali Barong | Bali | Bali |
| 2 | Bali_Merak | Bali Merak | Bali Peacock | Bali | Bali |
| 3 | Ceplok | Ceplok | Ceplok Geometric | Jawa Tengah | Jawa |
| 4 | Corak_Insang | Corak Insang | Pontianak Gill Pattern | Kalimantan Barat | Kalimantan |
| 5 | Ikat_Celup | Ikat Celup | Tie-Dye Batik | Nusantara | Nusantara |
| 6 | Jakarta_Ondel_Ondel | Jakarta Ondel Ondel | Jakarta Ondel-Ondel | DKI Jakarta | Jawa |
| 7 | Jawa_Barat_Megamendung | Jawa Barat Megamendung | Cirebon Cloud (Megamendung) | Jawa Barat | Jawa |
| 8 | Jawa_Timur_Pring | Jawa Timur Pring | East Java Bamboo (Pring Sedapur) | Jawa Timur | Jawa |
| 9 | Kalimantan_Dayak | Kalimantan Dayak | Dayak Tree of Life | Kalimantan | Kalimantan |
| 10 | Lampung_Gajah | Lampung Gajah | Lampung Elephant | Lampung | Sumatera |
| 11 | Lasem | Lasem | Lasem Sino-Javanese | Jawa Tengah | Jawa |
| 12 | Madura_Mataketeran | Madura Mataketeran | Madura Dove Eye | Madura | Madura |
| 13 | Maluku_Pala | Maluku Pala | Maluku Nutmeg Spice | Maluku | Maluku |
| 14 | NTB_Lumbung | NTB Lumbung | Lombok Rice Barn (Lumbung) | Nusa Tenggara Barat | Nusa Tenggara |
| 15 | Papua_Asmat | Papua Asmat | Papua Asmat Woodcarving | Papua | Papua |
| 16 | Papua_Cendrawasih | Papua Cendrawasih | Papua Bird of Paradise | Papua | Papua |
| 17 | Papua_Tifa | Papua Tifa | Papua Tifa Drum | Papua | Papua |
| 18 | Priangan_Merak_Ngibing | Priangan Merak Ngibing | Priangan Dancing Peacock | Jawa Barat | Jawa |
| 19 | Sekar | Sekar | Sekar Jagad Floral Universe | Jawa | Jawa |
| 20 | Sidoluhur | Sidoluhur | Sidoluhur Noble Attainment | Jawa Tengah / Yogyakarta | Jawa |
| 21 | Sogan | Sogan | Solo Classic Golden-Brown Sogan | Solo / Surakarta | Jawa |
| 22 | Solo_Parang | Solo Parang | Solo Parang Wave | Solo / Surakarta | Jawa |
| 23 | Sulawesi_Selatan_Lontara | Sulawesi Selatan Lontara | South Sulawesi Lontara Script | Sulawesi Selatan | Sulawesi |
| 24 | Sumatera_Barat_Rumah_Minang | Sumatera Barat Rumah Minang | West Sumatra Minang House | Sumatera Barat | Sumatera |
| 25 | Sumatera_Utara_Boraspati | Sumatera Utara Boraspati | North Sumatra Boraspati Lizard | Sumatera Utara | Sumatera |
| 26 | Tambal | Tambal | Tambal Patchwork Healing | Jawa | Jawa |
| 27 | Yogyakarta_Kawung | Yogyakarta Kawung | Yogyakarta Aren Palm (Kawung) | Yogyakarta | Jawa |
| 28 | Yogyakarta_Parang | Yogyakarta Parang | Yogyakarta Parang Rusak | Yogyakarta | Jawa |

## Repository Structure

```
Batik-Vision-AI/
├── web/                             # Production Next.js 16 client application
│   ├── public/
│   │   ├── models/                  # ONNX runtime models (batik_model_v3.onnx)
│   │   └── motifs/                  # Exemplar motif swatch imagery
│   ├── src/
│   │   ├── app/                     # Next.js layout and entrypage
│   │   ├── components/              # UI components (Scanner, Atlas, History, About)
│   │   ├── lib/                     # ONNX runner, YOLO detector, i18n dictionary
│   │   └── types/                   # TypeScript interfaces
│   ├── package.json
│   └── tsconfig.json
├── src/
│   └── ml/                          # Core ML pipeline modules
│       ├── pipeline.py              # EfficientNetV2S architecture, cosine warmup & capped weights
│       ├── losses.py                # Calibrated CCE + Categorical focal loss
│       ├── augmentations.py         # Domain-safe uint8 Albumentations pipeline
│       └── calibration.py           # Temperature scaling, ECE & OOD detector
├── notebooks/
│   ├── batik_motifs_v3_train.ipynb  # Modular training notebook
│   └── cells_v3/                    # Python training cells
├── models/
│   ├── labels.json                  # Comprehensive multilingual labels registry
│   └── labels.txt                   # Ordered 28-class list
├── results/
│   ├── v2.5/                        # Historical v2.5 evaluation baseline
│   └── v3_dataset_audit.csv         # Perceptual hash deduplication audit
├── scripts/                         # Leakage audit, headless Colab v4 trainer & ONNX export/parity
├── tests/                           # Pytest automated ML test suite
├── .github/workflows/ci.yml         # Python 3.11 + Node 22 GitHub Actions CI
├── docs/                            # Architectural specifications and plans
├── archive/
│   ├── v1/                          # Legacy v1 model artifacts
│   ├── v2_dev/                      # Legacy v2 development logs
│   └── v2_legacy/                   # Archived v2 Streamlit app, notebooks, and reports
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

### 1. Web Application (Next.js)

To run the interactive web application locally:

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

To generate an optimized production build:

```bash
cd web
npm run build
npm run start
```

### 2. Python Testing and Dataset Scripts

Set up the Python environment:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Run test suite:

```bash
pytest tests/ -v
```

## Model Training and Export

Training can be executed headlessly on Google Colab GPU (`T4` / `L4` / `A100`) or Kaggle using `scripts/train_v4_colab.py` (or `notebooks/batik_motifs_v3_train.ipynb`):

```bash
# 1. Install dev/training dependencies
pip install -r requirements-dev.txt

# 2. Fast 1-epoch smoke test on Colab T4 (auto-stops VM on exit)
colab run --gpu T4 --timeout 0 scripts/train_v4_colab.py --dataset-dir /content/dataset --smoke-test

# 3. Full 5-fold GroupStratifiedKFold training + FP16 ONNX export + calibration
python scripts/train_v4_colab.py --dataset-dir /content/dataset --backbone efficientnetv2s --img-size 300 --quantize fp16

# 4. Standalone Keras -> ONNX export (opset 17, optional FP16/INT8 quantization) & parity check
python scripts/convert_keras_to_onnx.py --input path/to/best_model.keras --output web/public/models/batik_model_v3.onnx --quantize fp16
python scripts/verify_onnx_parity.py --keras path/to/best_model.keras --onnx web/public/models/batik_model_v3.onnx
```

## License

This project is licensed under the MIT License. See LICENSE for details.

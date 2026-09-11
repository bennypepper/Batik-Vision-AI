# Batik Vision AI: Two-Stage Indonesian Batik Motifs Recognition Engine

Batik Vision AI is a computer vision platform designed to preserve, document, and classify 28 traditional Indonesian batik motifs. The system combines fabric localization via YOLO with high-resolution texture classification via EfficientNetV2S, executed entirely client-side using WebAssembly and WebGL.

## System Architecture

```
                       [ Input Image: Textile or Garment ]
                                        │
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │   Stage 1: Fabric Localization (YOLO)    │
                   │  - ROI detection and framing guidance    │
                   │  - Boundary edge gradient energy test    │
                   │  - Full Swatch vs Focus Center presets   │
                   └────────────────────┬─────────────────────┘
                                        │
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │ Stage 2: Motif Texture Classifier (CNN)  │
                   │  - Backbone: Fine-Tuned EfficientNetV2S  │
                   │  - Input: 224x224x3 float32 [0, 255]     │
                   │  - Internal Rescaling: [-1, 1]           │
                   │  - Output: 28-class probability vector   │
                   └────────────────────┬─────────────────────┘
                                        │
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │    Zero-Backend Client-Side Execution    │
                   │  - ONNX Runtime WebAssembly & WebGL      │
                   │  - Zero server transfer / complete privacy│
                   │  - Local history & Cultural Atlas viewer │
                   └──────────────────────────────────────────┘
```

## Architectural Improvements: v2 to v3

### 1. Elimination of Double Preprocessing
In v2, input images underwent double normalization: the preprocessing function scaled raw pixels from `[0, 255]` to `[-1, 1]`, and the Keras model internally applied a second normalization layer, compressing dynamic range to `[-1.008, -0.992]`. In v3, `EfficientNetV2S` is instantiated with `include_preprocessing=False` and a single, explicit internal `Rescaling(scale=1.0/127.5, offset=-1.0)` layer that accepts raw `[0, 255]` float32 tensors.

### 2. Perceptual Hash Deduplication
A comprehensive audit using 64-bit difference hashing (dHash with Hamming distance threshold <= 4) identified 799 duplicate and near-duplicate image pairs across splits and folds in the dataset. Deduplication eliminated data leakage between training and validation sets, ensuring valid generalization metrics.

### 3. Class Imbalance Handling: Focal Loss and Sample Weighting
Minority classes (such as Priangan Merak Ngibing and Sogan) suffered from high error rates in v2. Version 3 incorporates:
- Categorical Focal Loss with focusing parameter gamma = 2.0.
- Effective Number of Samples weighting based on (1 - beta) / (1 - beta^n) with beta = 0.999.

### 4. Two-Stage Fabric Localization Pipeline
Rather than forcing raw uncurated photographs directly into a texture classifier, Stage 1 uses YOLO fabric detection to isolate textile regions and calculate border spatial gradient energy. Close-up fabric swatches are automatically framed at 100% full pattern, while garments on models or mannequins default to 50% central focus with manual ROI fine-tuning capabilities.

### 5. Client-Side WebAssembly Execution
The trained EfficientNetV2S model is exported to ONNX format with FP16 weights (`batik_model_v3.onnx`) and evaluated directly inside the browser using ONNX Runtime Web. User photos are never uploaded to remote servers, eliminating hosting infrastructure costs and ensuring data privacy.

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
│       ├── pipeline.py              # EfficientNetV2S architecture and weighting
│       └── losses.py                # Categorical focal loss implementation
├── notebooks/
│   ├── batik_motifs_v3_train.ipynb  # End-to-end Kaggle training notebook
│   └── cells_v3/                    # Modular Python training cells
├── models/
│   ├── labels.json                  # Comprehensive multilingual labels registry
│   └── labels.txt                   # Ordered class list
├── results/
│   └── v3_results/                  # Kaggle evaluation charts, reports, and weights
├── scripts/                         # Dataset hygiene and ONNX conversion utilities
├── tests/                           # Pytest automated test suite
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

Training was performed on a Kaggle environment with an NVIDIA Tesla P100 GPU using 5-Fold Stratified Cross-Validation. To replicate training:

1. Import `notebooks/batik_motifs_v3_train.ipynb` into Kaggle.
2. Attach the dataset containing 28 batik motif classes.
3. Execute the notebook to produce `.keras`, `.tflite`, and `.onnx` artifacts.
4. Convert model weights to ONNX format using `scripts/convert_keras_to_onnx.py`.

## License

This project is licensed under the MIT License. See LICENSE for details.

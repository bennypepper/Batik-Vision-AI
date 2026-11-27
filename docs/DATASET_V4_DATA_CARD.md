# Indonesian Batik Dataset v2.0 — Leak-Free, DINOv2 + SigLIP Curated 28-Class Benchmark

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
**`3,470` Clean Full-Resolution Images · `28` Canonical Motif Classes · `3,140` Independent Semantic Clusters · `177` OOD Calibration Negatives**

---

## Executive Summary: What Changed in v2.0?

Version 2.0 is a ground-up, data-centric overhaul of our v1 release (`2,216` raw images). While auditing **22 public Indonesian batik datasets (`22,077` raw images across Kaggle, Roboflow Universe, Hugging Face, Mendeley Data, GitHub, and Wikimedia Commons)**, we uncovered systemic data hygiene defects that silently inflate reported accuracy on almost every existing public batik benchmark:

| Metric | v1 Release (Raw) | v1 After Leak Audit | **v2.0 Curated Benchmark** |
|---|---:|---:|---:|
| Total Batik Images | 2,216 | 1,506 *(D4-unique)* | **3,470** *(full-res, leak-free)* |
| Classes with `>= 100` Clean Images | 5 / 28 *(inflated)* | 2 / 28 | **18 / 28** *(11 classes `>= 150`)* |
| Smallest Class (`Priangan_Merak_Ngibing` in v1) | 7–34 | 7 | **37** (`Sumatera_Utara_Boraspati`), **66** (`Priangan_Merak_Ngibing`) |
| Max Class Cap (Prevents Majority Collapse) | Uncapped (230) | Uncapped | **200** *(via DINOv2 k-center diversity sampling)* |
| Duplicate & Augmentation Leakage | ~28.2% intra-class dups | Stripped | **0%** *(MD5 + 8-way D4 dHash + DINOv2 `cos < 0.94`)* |
| Split-Safe Cluster Groups (`cluster_id`) | None | None | **3,140 clusters** (`DINOv2 cos >= 0.88`) in `metadata.csv` |
| Out-of-Distribution (OOD) Calibration Set | None | None | **177 open-licensed images** (incl. 48 *Songket / Tenun / Ulos*) |
| **Frozen DINOv2 5-Fold Leak-Free OOF Macro-F1** | *(leaked)* | 82.36% | **86.46%** (`87.44%` Accuracy, zero fine-tuning!) |

---

## Critical Empirical Findings Across Public Batik Datasets

During our audit of 22 public repositories, we documented five recurring failure modes that cause standard train/val/test splits to overestimate real-world accuracy by **+10% to +18%**:

1. **Roboflow Synthetic `.rf.<hash>` Augmentation Leakage (`2,999` copies caught):**
   Public Roboflow Universe exports (`Batik.v6i.multiclass`, `Batik Indonesia Skripsi`, `Batik Jawa Barat`, `Indonesia Batik Classification`) pre-bake 3× augmented copies of the same base photo (`<stem>.rf.<hash1>.jpg`, `<stem>.rf.<hash2>.jpg`). In `Batik_v6i_multiclass`, **`1,491 / 3,028` (49.2%)** were synthetic copies; in `Batik_Indonesia_Skripsi`, **`1,270 / 2,653` (47.9%)** were synthetic copies. Splitting them randomly puts rotated/cropped twins of test images directly into training!
2. **Severe Cross-Dataset Recycling (`90%–94%` overlap):**
   Many independently uploaded Kaggle and Hugging Face batik datasets recycle the exact same base web-scraped pool: `hf_salman_batik_indonesia` (`94.1%` overlap), `kaggle_hendryhb_batik_nusantara` (`90.6%` overlap), `kaggle_dwibudisantoso_wastra` & `kaggle_panduagas_batik_jawa_barat` (`100%` overlap after combining earlier sources). Naively concatenating public datasets creates massive hidden duplication.
3. **Orientation-Blind Perceptual Hashing Misses Flips & 90° Rotations:**
   Standard `dHash` misses horizontally flipped or 90°/180°/270° rotated augmentations. Using **8-way D4-dihedral `dHash <= 3`** plus **GPU `DINOv2-ViT-B/14` semantic deduplication (`cosine >= 0.94`)** caught an additional **`275` hidden intra-class semantic duplicates** (zoomed crops, slight perspective shifts, re-framed photos of the exact same cloth) that perceptual hashing alone missed.
4. **Cross-Class Label Conflicts & Non-Batik Web-Scrape Junk (`461` images removed):**
   - `303` cross-class dHash conflicts + `11` cross-class semantic duplicates (`cosine >= 0.94` with conflicting labels across datasets — e.g., *Kawung* misfiled inside *Kalimantan_Dayak* and *Ceplok*).
   - `103` non-batik junk images (portraits, webpage screenshots, solid plain garments, logos, text graphics) caught by `SigLIP-SO400M-384`.
   - `44` high-confidence mislabels caught by 5-fold leak-free Out-of-Fold (`OOF`) DINOv2 linear probing (`p(given_class) < 0.08` while `p(predicted_class) >= 0.75`).
5. **Ambiguous Umbrella Labels Quarantined (`3,396` images protected from blind mislabeling):**
   Generic folders such as `Parang` (`1,902` images), `Batik Papua` (`780`), `Batik Bali` (`370`), `Batik Madura` (`300`), and `Batik Priangan` (`100`) mix distinct motif classes together (e.g., *Solo Parang* vs *Yogyakarta Parang* differ in diagonal orientation and colorway; *Batik Papua* mixes *Cendrawasih*, *Asmat*, and *Tifa*). Blindly merging them corrupts fine-grained class boundaries.

---

## 5-Stage Curation Pipeline & Methodology

```
22 Public Sources (22,077 raw images: 2,189 v1 + 19,888 external)
   │
   ├─► Stage 1: Canonical 28-Class Taxonomy & Strict Ambiguity Quarantine
   │     • Unambiguous labels mapped cleanly; generic Parang/Papua/Bali/Betawi/Priangan/Madura quarantined
   │     • Single-class YOLO boxes cropped when bbox < 70% frame area (removes background clutter)
   │     • Low-res workshop patch folders (28x28, 142x142) excluded; min(w, h) >= 200px enforced for all new data
   │
   ├─► Stage 2: Multi-Layer Exact + Geometric Deduplication (CPU)
   │     • Roboflow `.rf.<hash>` stem pre-collapse (kept single highest-res representative; -2,999 copies)
   │     • Exact MD5 + 8-way D4-dihedral dHash (4 rotations × 2 flips, Hamming <= 3) across all 22 sources
   │     • Rejected 303 cross-class conflicts → 4,607 D4-unique unambiguous candidates + 1,112 unique ambiguous candidates
   │
   ├─► Stage 3: GPU Semantic Deduplication & Noise Filtering (DINOv2-ViT-B/14 + SigLIP-SO400M-384)
   │     • Full cosine similarity matrix on 768-d L2-normalized DINOv2 CLS embeddings
   │     • Intra-class connected components at cos >= 0.94 → keep single highest-quality representative (-275 dups)
   │     • Cross-class pairs at cos >= 0.94 → k-NN consensus adjudication (-11 cross-class dups)
   │     • SigLIP-SO400M positive textile vs negative junk prompt margin → drops non-batik junk (-103 images)
   │     • 5-fold leak-free StratifiedGroupKFold (clusters at cos >= 0.88) OOF probe → drops clear mislabels (-44 images)
   │     • OOF Macro-F1 jumped from 80.45% (pre-filter) → 86.46% (final)!
   │
   ├─► Stage 4: High-Precision Mining of Quarantined Ambiguous Pools (+23 verified images)
   │     • Strictly restricted to valid regional child classes (e.g., generic Parang → Solo_Parang only)
   │     • Required ALL 4 gates: top-1 prob >= 0.82, top1-top2 margin >= 0.35, class centroid sim >= 0.48, SigLIP > 0.0, cos < 0.94
   │     • Result: +21 Solo_Parang, +1 Madura_Mataketeran, +1 Papua_Tifa (zero noise pollution!)
   │
   └─► Stage 5: Diversity-Preserving Class Cap (max 200/class) & Leak-Free Cluster IDs
         • Over-represented classes (>200) downsampled via greedy furthest-point (k-center) sampling in DINOv2 space (-727 excess)
         • Final connected components at cos >= 0.88 exported as `cluster_id` (3,140 groups) in `metadata.csv`
         • Full-resolution EXIF-corrected RGB JPEGs assembled (median 320×320, p90 750×960, max 9148×6299)
```

---

## Complete 28-Class Breakdown & Frozen DINOv2 Baseline

*(Evaluated with a frozen `facebook/dinov2-base` linear probe under strict 5-fold `StratifiedGroupKFold` grouped by `cluster_id` — zero fine-tuning, zero data leakage.)*

| Class | From v1 Base | Net-New (External) | High-Precision Mined | **Final Count** | Frozen DINOv2 OOF F1 |
|---|---:|---:|---:|---:|---:|
| `Bali_Barong` | 35 | 85 | 0 | **120** | 97.1% |
| `Bali_Merak` | 31 | 14 | 0 | **45** | 69.3% |
| `Ceplok` | 26 | 174 | 0 | **200** | 79.3% |
| `Corak_Insang` | 52 | 148 | 0 | **200** | 97.3% |
| `Ikat_Celup` | 38 | 162 | 0 | **200** | 98.2% |
| `Jakarta_Ondel_Ondel` | 40 | 95 | 0 | **135** | 97.0% |
| `Jawa_Barat_Megamendung` | 55 | 145 | 0 | **200** | 97.5% |
| `Jawa_Timur_Pring` | 49 | 151 | 0 | **200** | 87.2% |
| `Kalimantan_Dayak` | 86 | 114 | 0 | **200** | 95.8% |
| `Lampung_Gajah` | 32 | 6 | 0 | **38** | 87.7% |
| `Lasem` | 52 | 120 | 0 | **172** | 72.2% |
| `Madura_Mataketeran` | 38 | 21 | 1 | **60** | 90.3% |
| `Maluku_Pala` | 31 | 79 | 0 | **110** | 84.6% |
| `NTB_Lumbung` | 32 | 9 | 0 | **41** | 88.1% |
| `Papua_Asmat` | 38 | 12 | 0 | **50** | 89.3% |
| `Papua_Cendrawasih` | 71 | 129 | 0 | **200** | 87.9% |
| `Papua_Tifa` | 37 | 12 | 1 | **50** | 91.7% |
| `Priangan_Merak_Ngibing` | 7 | 59 | 0 | **66** | 87.7% |
| `Sekar` | 45 | 120 | 0 | **165** | 77.7% |
| `Sidoluhur` | 34 | 112 | 0 | **146** | 84.1% |
| `Sogan` | 49 | 22 | 0 | **71** | 61.1% |
| `Solo_Parang` | 71 | 1 | 21 | **93** | 73.9% |
| `Sulawesi_Selatan_Lontara` | 34 | 98 | 0 | **132** | 92.0% |
| `Sumatera_Barat_Rumah_Minang` | 33 | 81 | 0 | **114** | 90.6% |
| `Sumatera_Utara_Boraspati` | 32 | 5 | 0 | **37** | 91.4% |
| `Tambal` | 46 | 65 | 0 | **111** | 90.0% |
| `Yogyakarta_Kawung` | 14 | 186 | 0 | **200** | 88.0% |
| `Yogyakarta_Parang` | 28 | 86 | 0 | **114** | 73.9% |
| **TOTAL (28 classes)** | **1136** | **2311** | **23** | **3470** | **86.46% Macro-F1** |

---

## Directory Layout & File Contents

```text
├── README.md                    # This full Data Card, findings, methodology & usage guide
├── LICENSE_AND_SOURCES.md       # Complete per-source attribution, URLs & licenses (all 21 sources + Wikimedia + OOD)
├── metadata.csv                 # Row-level provenance, cluster_id, dimensions, OOF prob & SigLIP score (3,470 rows)
├── labels.txt                   # Canonical 28-class list (sorted, 1 class per line)
├── curation_audit_summary.json  # Machine-readable audit funnel, per-class F1 & source counts
├── dataset/                     # 28 class folders containing 3,470 full-resolution RGB JPEGs
│   ├── Bali_Barong/
│   ├── ...
│   └── Yogyakarta_Parang/
└── ood_negatives/               # 177 license-verified OOD negative images across 6 buckets
    ├── everyday_objects/        # 27 images
    ├── indo_non_batik_textiles/ # 48 hard negatives (Songket, Tenun Ikat, Ulos)
    ├── indoor_textures/         # 50 images
    ├── non_batik_patterns/      # 40 images
    ├── plain_garments/          # 7 images
    ├── portraits_people/        # 5 images
    └── ood_manifest.csv         # Full Wikimedia title, license & URL attribution
```

---

## Recommended Training & Evaluation Best Practices

### 1. Always Split by `cluster_id` (Zero Near-Duplicate Leakage!)
Even after stripping `cosine >= 0.94` duplicates, different photos of the same batik cloth style can still share `0.88 <= cosine < 0.94`. Always split using `StratifiedGroupKFold` on `cluster_id`:

```python
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("metadata.csv")
sgkf = StratifiedGroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df["label"], groups=df["cluster_id"])):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    print(f"Fold {fold}: train={len(train_df)}, val={len(val_df)}, shared_clusters={len(set(train_df.cluster_id) & set(val_df.cluster_id))}")
```

### 2. Orientation-Sensitive Classes (Do NOT Blindly Horizontal-Flip!)
Three motifs in this taxonomy carry directional semantics:
- **`Solo_Parang`** vs **`Yogyakarta_Parang`** — diagonal parang blade slope is a key cultural discriminator; random horizontal flips invert one style into the other!
- **`Sulawesi_Selatan_Lontara`** — contains Bugis-Makassar Lontara script characters; horizontal flipping mirrors text.

Disable horizontal flips (and limit rotation to `<= 15°` and hue jitter to `<= 5°` to preserve *Sogan* / *Lasem* traditional dye colorways) for orientation-sensitive classes.

### 3. Calibrate Out-of-Distribution (OOD) Rejection With `ood_negatives/`
Real-world batik scanner apps receive non-batik fabrics (*Songket*, *Tenun*, *Ulos*), plain clothes, walls, and portraits. After training your classifier:
1. Fit **Temperature Scaling** ($T$) on your validation fold to minimize NLL / ECE.
2. Sweep Maximum Softmax Probability (`MSP`) or Energy thresholds separating your validation batik fold from `ood_negatives/` (targeting `>= 90%` OOD rejection at `<= 5%` in-distribution false rejection).

---

## Sources & Attribution

This benchmark aggregates, deduplicates, cleans, and harmonizes images from **21 open datasets and archives** across Kaggle, Roboflow Universe, Mendeley Data, Hugging Face, GitHub, and Wikimedia Commons. See **`LICENSE_AND_SOURCES.md`** for the complete per-source citation table, URLs, and license breakdown.

| Source ID | Dataset / Archive | Platform | License | Images Retained | Notes |
|---|---|---|---|---:|---|
| `existing_local` | [v1 Base Release (fisheightcharacter / Batik-Indonesia by M. Salman Al Faridzi)](https://www.kaggle.com/datasets/fisheightcharacter/indonesian-batik-dataset-enhanced-and-cleaned) | Kaggle / Hugging Face | Apache-2.0 | **1136** | 28-class v1 base dataset derived from Hugging Face Batik-Indonesia + manually verified v1 expansions (deduplicated & noise-filtered). |
| `kaggle_buyungsaloka_motif_batik` | [motif-batik-dataset (buyungsaloka)](https://www.kaggle.com/datasets/buyungsaloka/motif-batik-dataset) | Kaggle | Unlisted on Kaggle (attributed; verify upstream terms for commercial use) | **608** | Major contributor for regional motifs: Bali_Barong, Sulawesi_Selatan_Lontara, Madura_Mataketeran, Jakarta_Ondel_Ondel, Sumatera_Barat_Rumah_Minang, Jawa_Timur_Pring. |
| `kaggle_alfanme_corak_app` | [Indonesian Batik Motifs (Corak App) (alfanme)](https://www.kaggle.com/datasets/alfanme/indonesian-batik-motifs-corak-app) | Kaggle | CC0-1.0 | **420** | Unambiguous motifs mapped (Cendrawasih, Dayak, Ikat Celup, Insang, Kawung, Lasem, Megamendung, Pala, Sekar Jagad, Tambal); generic Parang/Bali/Betawi excluded. |
| `Batik_v6i_multiclass` | [Batik Multiclass v6 (Roboflow Universe — a-co6il/batik-ifddz)](https://universe.roboflow.com/a-co6il/batik-ifddz) | Roboflow Universe | CC BY 4.0 | **296** | 1,491 synthetic .rf. augmentation duplicates stripped prior to inclusion; mapped Ikat+Celup, Corak+Insang, Pring+Sedapur, Dayak, Kawung, Megamendung. |
| `Batik_Indonesia_Skripsi_v52i_yolov8` | [Batik Indonesia Skripsi v52 (Roboflow Universe — aku-cinta-indonesia/batik-indonesia-skripsi)](https://universe.roboflow.com/aku-cinta-indonesia/batik-indonesia-skripsi) | Roboflow Universe | CC BY 4.0 | **241** | 1,270 synthetic .rf. augmentation duplicates stripped; single-class tight bounding boxes cropped when bbox < 70% frame area. |
| `kaggle_sendytegar_batik_keraton` | [Dataset Batik Keraton (sendytegar)](https://www.kaggle.com/datasets/sendytegar/dataset-batik-keraton) | Kaggle | CC0-1.0 | **159** | Contributed Kawung, Megamendung, and high-precision DINOv2-mined Solo_Parang images. |
| `Indonesia_Batik_v7_folder` | [Indonesia Batik Classification v7 (Roboflow Universe — melanie-gabriela-tjandrasubrata)](https://universe.roboflow.com/melanie-gabriela-tjandrasubrata/indonesia-batik-classification-mktx6) | Roboflow Universe | CC BY 4.0 | **118** | 112 synthetic .rf. duplicates stripped; mapped Kawung, Mega-Mendung, Sekar-Jagad, Sidoluhur. |
| `gh_dionisiusfb_batik` | [batik-motif-classification (DionisiusFB)](https://github.com/DionisiusFB/batik-motif-classification) | GitHub | Public GitHub Repository (academic) | **110** | Contributed Ceplok and Kawung motifs. |
| `kaggle_dionisiusdh_batik_motifs` | [Indonesian Batik Motifs (dionisiusdh)](https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs) | Kaggle | CC0-1.0 | **97** | Contributed Ceplok, Sogan, Sidoluhur, Lasem, Tambal, Sekar, Ikat_Celup, Cendrawasih; generic Bali/Betawi/Parang/Priangan excluded. |
| `kaggle_brendhaadiyanvianca_batik_indonesian` | [Batik Indonesian (brendhaadiyanvianca)](https://www.kaggle.com/datasets/brendhaadiyanvianca/batik-indonesian) | Kaggle | Apache-2.0 | **78** | Explicit Yogyakarta_Parang, JawaTengah_Lasem, Kalimantan_Insang, Yogyakarta_SekarJagad, Yogyakarta_Tambal. |
| `kaggle_syahdanputra_batiksnap` | [BatikSnap Dataset (syahdanputra)](https://www.kaggle.com/datasets/syahdanputra/batiksnap-dataset) | Kaggle | DbCL-1.0 | **62** | Contributed Insang, Kawung, Megamendung, Sidoluhur. |
| `Batik_Jawa_Barat_v6-5classes_yolov8` | [Batik Jawa Barat v6 (Roboflow Universe — batikjabar/batik-jawa-barat)](https://universe.roboflow.com/batikjabar/batik-jawa-barat) | Roboflow Universe | CC BY 4.0 | **50** | 126 synthetic .rf. duplicates stripped; primary source for Priangan_Merak_Ngibing expansion (+39 net-new) + Megamendung. |
| `Mendeley_Lasem_GunungRinggit_Originals` | [Batik Lasem Motifs Collections (Gunung Ringgit, Kricak, Latohan, Nyuk Pitu, Seritan) — Full-Res Originals](https://data.mendeley.com/datasets/kw5b7d9zgn) | Mendeley Data | CC BY 4.0 | **31** | ONLY full-resolution original workshop textile photographs retained (tiny 28x28 and 142x142 isen-isen patches excluded). |
| `wikimedia_batik` | [Wikimedia Commons Museum Textile Archives (Collectie NMvWereldculturen & open-licensed batik)](https://commons.wikimedia.org/) | Wikimedia Commons | Public Domain / CC BY-SA 4.0 | **25** | High-resolution museum & archival batik photos (Sekar Jagad, Yogyakarta Parang, Megamendung, Papua Cendrawasih). |
| `kaggle_fisheightcharacter_batik_enhanced` | [Indonesian Batik Dataset (Enhanced & Cleaned) v1 Delta](https://www.kaggle.com/datasets/fisheightcharacter/indonesian-batik-dataset-enhanced-and-cleaned) | Kaggle | Apache-2.0 | **20** | Net-new unique Priangan_Merak_Ngibing images present in the v1 Kaggle release. |
| `hf_salman_batik_indonesia` | [Batik-Indonesia (Muhammad Salman Al Faridzi)](https://huggingface.co/datasets/muhammadsalmanalfaridzi/Batik-Indonesia) | Hugging Face | Apache-2.0 | **7** | Original upstream backbone for v1; 7 additional non-overlapping images retained after deduplication. |
| `kaggle_hendryhb_batik_nusantara` | [Batik Nusantara (Batik Indonesia) Dataset (hendryhb)](https://www.kaggle.com/datasets/hendryhb/batik-nusantara-batik-indonesia-dataset) | Kaggle | CC0-1.0 | **6** | 90.6% overlap with v1 base removed; 6 net-new clean images retained. |
| `gh_reyyichiro_batik` | [batik-motif-classification (ReyyIchiro)](https://github.com/ReyyIchiro/batik-motif-classification) | GitHub | Public GitHub Repository (academic) | **3** | 3 high-precision DINOv2-mined Solo_Parang images retained after strict dedup. |
| `mendeley_lasem_kw5b7d9zgn` | [Batik Lasem Motifs Collections (Mendeley API mirror kw5b7d9zgn)](https://data.mendeley.com/datasets/kw5b7d9zgn) | Mendeley Data | CC BY 4.0 | **1** | 1 full-res Lasem textile photograph passing >=200px & dedup gates. |
| `kaggle_yohanpermanautm_batik_madura` | [Citra Batik Madura & Flora (yohanpermanautm)](https://www.kaggle.com/datasets/yohanpermanautm/citra-batik-madura-and-flora) | Kaggle | Apache-2.0 | **1** | 1 high-precision DINOv2-mined Madura_Mataketeran image passing >=0.82 prob & >=0.35 margin gate. |
| `kaggle_fiin28_batik_papua` | [Batik Papua (fiin28)](https://www.kaggle.com/datasets/fiin28/batik-papua) | Kaggle | MIT | **1** | 1 high-precision DINOv2-mined Papua_Tifa image passing >=0.82 prob & >=0.35 margin gate. |

---

## Changelog
- **v2.0 (Current):** Expanded from `1,506` clean unique (`2,216` raw) → **`3,470` full-resolution leak-free images** (`18/28` classes `>= 100`, capped at `200/class` via DINOv2 k-center diversity sampling); audited 22 public sources; removed `2,999` Roboflow `.rf.` synthetic duplicates, `275` semantic duplicates, `11` cross-class duplicates, `103` non-batik junk images, and `44` mislabels; added `metadata.csv` with `3,140` leak-free `cluster_id`s, `177` open-licensed OOD calibration negatives (including 48 Indonesian non-batik textiles), and full provenance/licensing documentation (`README.md`, `LICENSE_AND_SOURCES.md`, `curation_audit_summary.json`).
- **v1.0:** Initial 28-class cleaned subset (`2,216` images) derived from `Batik-Indonesia` (Apache-2.0) with manual expansions for *Sogan*, *Lasem*, *Ceplok*, and *Priangan_Merak_Ngibing*.

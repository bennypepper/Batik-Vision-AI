# Batik Motifs v3 Phase 1 Implementation Plan: Dataset Hygiene & Pipeline Refactoring

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish dataset integrity, perceptual deduplication, canonical label registries, and build the bug-free v3 training pipeline module (with the double-preprocessing defect resolved and Focal Loss implemented).

**Architecture:** A modular Python machine learning toolkit with automated dataset hashing, schema-validated JSON label registries, and an isolated `src/ml/` package providing the corrected Keras data loaders, Focal Loss function, and ONNX conversion pipelines.

**Tech Stack:** Python 3.10/3.11, TensorFlow/Keras, ImageHash, Pillow, OpenCV, NumPy, Scikit-Learn, Pytest.

**Spec:** `docs/specs/2026-09-11-batik-cnn-v3-overhaul-design.md`

## Global Constraints

- Strictly ZERO emojis across code, comments, documentation, and git commit messages.
- Strictly ZERO em dashes (`—`) across code, comments, documentation, and git commit messages.
- Local git commits only; never push remotely without explicit user confirmation.
- TDD workflow: Write failing test first, verify failure, implement minimal code, verify pass, commit.

---

### Task 1: Canonical Label and Cultural Metadata Registry

**Files:**
- Create: `scripts/generate_labels_registry.py`
- Create: `models/labels.json`
- Test: `tests/test_labels_registry.py`

**Interfaces:**
- Consumes: `src/utils/data.py` (REGION_MAP, BATIK_INFO, WEAK_CLASSES) and `models/labels.txt`
- Produces: `models/labels.json` schema:
  ```json
  {
    "classes": [
      {
        "id": 0,
        "slug": "Bali_Barong",
        "name_id": "Bali Barong",
        "name_en": "Bali Barong",
        "region": "Bali",
        "island": "Bali",
        "is_weak_class": false,
        "description_id": "...",
        "description_en": "..."
      }
    ],
    "total_classes": 28
  }
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labels_registry.py
import json
import os
import pytest
from scripts.generate_labels_registry import build_labels_registry

def test_labels_registry_structure(tmp_path):
    output_json = tmp_path / "labels.json"
    registry = build_labels_registry(output_path=str(output_json))
    
    assert registry["total_classes"] == 28
    assert len(registry["classes"]) == 28
    
    first = registry["classes"][0]
    assert "id" in first
    assert "slug" in first
    assert "name_id" in first
    assert "region" in first
    assert "description_id" in first
    assert isinstance(first["is_weak_class"], bool)
    assert os.path.exists(output_json)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_labels_registry.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement minimal code**

Implement `scripts/generate_labels_registry.py` reading from `models/labels.txt` and `src/utils/data.py`, then generate `models/labels.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_labels_registry.py -v`
Expected: PASS

- [ ] **Step 5: Run generator to create production `models/labels.json` and commit**

Run: `python scripts/generate_labels_registry.py`
Commit: `git add models/labels.json scripts/generate_labels_registry.py tests/test_labels_registry.py && git commit -m "feat(ml): generate canonical 28-class metadata registry"`

---

### Task 2: Perceptual Hash (pHash) Dataset Leakage and Deduplication Audit

**Files:**
- Create: `scripts/audit_dataset_leakage.py`
- Test: `tests/test_audit_dataset_leakage.py`

**Interfaces:**
- Consumes: Target dataset directory path (e.g. `D:\Personal Projects\Computer Vision-ML Projects\Batik Motifs Detector v2\Batik_dataset\dataset`)
- Produces: `scan_dataset_duplicates(dataset_dir: str, threshold: int = 3) -> tuple[dict, list[dict]]`
  Returns dictionary of file hashes and list of flagged duplicate pairs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audit_dataset_leakage.py
import os
from PIL import Image
import numpy as np
import pytest
from scripts.audit_dataset_leakage import compute_image_phash, scan_dataset_duplicates

def test_phash_computation(tmp_path):
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img1_path)
    Image.fromarray(arr).save(img2_path)
    
    h1 = compute_image_phash(str(img1_path))
    h2 = compute_image_phash(str(img2_path))
    
    assert h1 == h2

def test_scan_dataset_duplicates(tmp_path):
    c1 = tmp_path / "ClassA"
    c2 = tmp_path / "ClassB"
    c1.mkdir()
    c2.mkdir()
    
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img_a = c1 / "a.jpg"
    img_b = c2 / "b.jpg"
    Image.fromarray(arr).save(img_a)
    Image.fromarray(arr).save(img_b)
    
    hashes, duplicates = scan_dataset_duplicates(str(tmp_path), threshold=0)
    assert len(duplicates) == 1
    assert duplicates[0]["distance"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audit_dataset_leakage.py -v`
Expected: FAIL

- [ ] **Step 3: Implement minimal code**

Implement `scripts/audit_dataset_leakage.py` using `imagehash.phash` or pure PIL/NumPy difference hashing (dHash) to avoid heavy external dependencies.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_audit_dataset_leakage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

Commit: `git add scripts/audit_dataset_leakage.py tests/test_audit_dataset_leakage.py && git commit -m "feat(dataset): add perceptual hash leakage and duplicate audit tool"`

---

### Task 3: Bug-Free V3 Preprocessing Pipeline & Focal Loss Module

**Files:**
- Create: `src/ml/__init__.py`
- Create: `src/ml/pipeline.py`
- Create: `src/ml/losses.py`
- Test: `tests/test_ml_pipeline.py`

**Interfaces:**
- Produces:
  * `create_v3_backbone_model(num_classes: int = 28, input_shape=(224, 224, 3)) -> tuple[tf.keras.Model, tf.keras.Model]`
    Guarantees: Input accepts raw $[0, 255]$ pixels, `include_preprocessing=False`, internal `Rescaling(scale=1/127.5, offset=-1.0)` prepended.
  * `CategoricalFocalLoss(gamma: float = 2.0, label_smoothing: float = 0.1)`
  * `compute_effective_sample_weights(class_counts: list[int], beta: float = 0.999) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ml_pipeline.py
import numpy as np
import tensorflow as tf
import pytest
from src.ml.pipeline import create_v3_backbone_model, compute_effective_sample_weights
from src.ml.losses import CategoricalFocalLoss

def test_v3_model_rescaling_input():
    base_model, model = create_v3_backbone_model(num_classes=28)
    
    # Test that model accepts raw [0, 255] float input without errors
    dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)
    output = model(dummy_input, training=False)
    
    assert output.shape == (1, 28)
    np.testing.assert_almost_equal(tf.reduce_sum(output, axis=-1).numpy(), [1.0], decimal=5)

def test_effective_sample_weights():
    counts = [34, 100, 230]
    weights = compute_effective_sample_weights(counts, beta=0.999)
    assert len(weights) == 3
    # Minority class should have highest weight
    assert weights[0] > weights[1] > weights[2]

def test_focal_loss_computation():
    loss_fn = CategoricalFocalLoss(gamma=2.0, label_smoothing=0.1)
    y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8]], dtype=tf.float32)
    loss = loss_fn(y_true, y_pred)
    assert float(loss) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement minimal code**

Implement `src/ml/pipeline.py` and `src/ml/losses.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

Commit: `git add src/ml/pipeline.py src/ml/losses.py tests/test_ml_pipeline.py && git commit -m "feat(ml): implement v3 bug-free model pipeline and focal loss"`

---

### Task 4: Modular Retraining Orchestrator and Notebook Refactor

**Files:**
- Create: `notebooks/batik_motifs_v3_train.ipynb`
- Create: `notebooks/cells_v3/`
- Test: `tests/test_v3_pipeline_integration.py`

**Interfaces:**
- Produces: Complete, self-contained Kaggle/Colab notebook and Python executable scripts implementing:
  1. Config & Deterministic Seed.
  2. Dataset loading from `D:\...\Batik_dataset\dataset` or Kaggle input.
  3. Stratified 5-Fold split.
  4. Phase 1 & Phase 2 training with Focal Loss and Cosine Warmup.
  5. Automated conversion to ONNX via `tf2onnx` and FP16 quantization.

- [ ] **Step 1: Write integration test**

```python
# tests/test_v3_pipeline_integration.py
import pytest
from src.ml.pipeline import create_v3_backbone_model
from src.ml.losses import CategoricalFocalLoss

def test_v3_compilation_and_step():
    base, model = create_v3_backbone_model(num_classes=5)
    model.compile(
        optimizer="adam",
        loss=CategoricalFocalLoss(gamma=2.0),
        metrics=["accuracy"]
    )
    assert model.loss is not None
```

- [ ] **Step 2: Run test to verify passes**

Run: `pytest tests/test_v3_pipeline_integration.py -v`
Expected: PASS

- [ ] **Step 3: Assemble clean `notebooks/batik_motifs_v3_train.ipynb` and export cells**

Integrate the validated pipeline into the full training notebook.

- [ ] **Step 4: Commit**

Commit: `git add notebooks/ tests/test_v3_pipeline_integration.py && git commit -m "feat(notebooks): assemble complete v3 training notebook and modular cells"`

---

## Plan Verification Checklist

- [ ] All 4 tasks have unambiguous file targets and interfaces.
- [ ] Strictly zero emojis and zero em dashes across all code and documentation.
- [ ] Every step has clear commands and expected outputs.
- [ ] Complete test suite passes locally with `pytest tests/ -v`.

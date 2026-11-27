"""TDD hardening tests for v4 ML pipeline remediations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image


def _create_rgb_image(path: Path, width: int = 512, height: int = 384, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


class TestDomainSafeAugmentations:
    """P0-2 & P1-2: uint8 Albumentations and Batik domain-safe constraints."""

    def test_load_image_uint8_preserves_native_resolution_and_dtype(self, tmp_path: Path) -> None:
        from src.ml.augmentations import load_image_uint8

        img_path = _create_rgb_image(tmp_path / "hi_res.png", width=640, height=480, seed=1)
        loaded = load_image_uint8(str(img_path))

        assert loaded.dtype == np.uint8
        assert loaded.shape == (480, 640, 3)

    def test_uint8_augmentation_produces_valid_float32_0_255_no_nans(self, tmp_path: Path) -> None:
        from src.ml.augmentations import augment_image_file

        img_path = _create_rgb_image(tmp_path / "batik.png", width=512, height=512, seed=2)
        for tier in ("tier1", "tier2", "tier3"):
            out = augment_image_file(
                str(img_path),
                class_name="Ceplok",
                img_size=300,
                tier=tier,
                training=True,
            )
            assert out.dtype == np.float32
            assert out.shape == (300, 300, 3)
            assert not np.isnan(out).any()
            assert not np.isinf(out).any()
            assert float(out.min()) >= 0.0
            assert float(out.max()) <= 255.0
            # Ensure image wasn't collapsed to near-black [0, 1] by float32 Albumentations bug
            assert float(out.max()) > 50.0

    def test_domain_destructive_transforms_removed_and_hsv_bounded(self) -> None:
        from src.ml.augmentations import build_train_augmentation

        for tier in ("tier1", "tier2", "tier3"):
            pipeline = build_train_augmentation(img_size=224, tier=tier, allow_horizontal_flip=True)
            transform_types = {type(t) for t in pipeline.transforms}
            assert A.GridDistortion not in transform_types
            assert A.ElasticTransform not in transform_types

            for t in pipeline.transforms:
                if isinstance(t, A.HueSaturationValue):
                    # Albumentations stores limits as tuples (-limit, +limit)
                    hue_max = max(abs(x) for x in t.hue_shift_limit)
                    sat_max = max(abs(x) for x in t.sat_shift_limit)
                    val_max = max(abs(x) for x in t.val_shift_limit)
                    assert hue_max <= 5
                    assert sat_max <= 15
                    assert val_max <= 15

    def test_orientation_sensitive_classes_disable_horizontal_flip(self) -> None:
        from src.ml.augmentations import (
            ORIENTATION_SENSITIVE_CLASSES,
            build_train_augmentation,
            is_orientation_sensitive,
        )

        expected = {"Solo_Parang", "Yogyakarta_Parang", "Sulawesi_Selatan_Lontara"}
        assert expected.issubset(ORIENTATION_SENSITIVE_CLASSES)

        for cls_name in expected:
            assert is_orientation_sensitive(cls_name) is True
            pipeline = build_train_augmentation(
                img_size=224,
                tier="tier1",
                allow_horizontal_flip=not is_orientation_sensitive(cls_name),
            )
            assert not any(isinstance(t, A.HorizontalFlip) for t in pipeline.transforms)

        assert is_orientation_sensitive("Ceplok") is False
        ceplok_pipeline = build_train_augmentation(img_size=224, tier="tier1", allow_horizontal_flip=True)
        assert any(isinstance(t, A.HorizontalFlip) for t in ceplok_pipeline.transforms)


class TestLossesAndPipelineHardening:
    """P1-1: Loss calibration, capped effective weights, phase-2 unfreeze, cosine warmup."""

    def test_effective_class_weights_capped_at_max_weight(self) -> None:
        from src.ml.pipeline import compute_effective_class_weights

        counts = [5, 10, 50, 200, 1000]
        weights = compute_effective_class_weights(counts, beta=0.999, max_weight=3.0)

        assert len(weights) == len(counts)
        assert float(np.max(weights)) <= 3.0 + 1e-6
        assert weights[0] >= weights[1] > weights[2] > weights[3] > weights[4]

    def test_build_classification_loss_modes(self) -> None:
        from src.ml.losses import CategoricalFocalLoss, build_classification_loss

        cce = build_classification_loss(mode="cce_smooth", label_smoothing=0.05)
        assert isinstance(cce, tf.keras.losses.CategoricalCrossentropy)

        focal = build_classification_loss(mode="focal", label_smoothing=0.0, gamma=2.0)
        assert isinstance(focal, CategoricalFocalLoss)

        with pytest.raises(ValueError, match="Unsupported loss mode"):
            build_classification_loss(mode="invalid_loss_mode")

    def test_focal_loss_uses_true_class_pt_for_modulating_factor(self) -> None:
        from src.ml.losses import CategoricalFocalLoss

        # With label_smoothing=0.0, true-class modulating factor (1 - p_t)^gamma must match manual formula
        loss_fn = CategoricalFocalLoss(gamma=2.0, label_smoothing=0.0)
        y_true = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.8, 0.1, 0.1]], dtype=tf.float32)

        actual = float(loss_fn(y_true, y_pred))
        expected = -((1.0 - 0.8) ** 2.0) * np.log(0.8)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

        # With label_smoothing=0.1, modulating factor must still scale by (1 - p_t)^gamma
        # where p_t = 0.8 (so (1 - 0.8)^2 = 0.04), NOT inflate off-target classes by (1 - 0.1)^2 = 0.81
        loss_smooth = CategoricalFocalLoss(gamma=2.0, label_smoothing=0.1)
        val_smooth = float(loss_smooth(y_true, y_pred))
        y_smooth = np.array([0.9 + 0.1 / 3.0, 0.1 / 3.0, 0.1 / 3.0])
        expected_smooth = float(-((1.0 - 0.8) ** 2.0) * np.sum(y_smooth * np.log([0.8, 0.1, 0.1])))
        np.testing.assert_allclose(val_smooth, expected_smooth, rtol=1e-4)

    def test_configurable_img_size_phase2_unfreeze_and_cosine_warmup(self) -> None:
        from src.ml.pipeline import (
            build_cosine_warmup_schedule,
            configure_phase2_unfreeze,
            create_v3_backbone_model,
        )

        base_model, model = create_v3_backbone_model(num_classes=5, img_size=300, weights=None)
        assert model.input_shape == (None, 300, 300, 3)

        trainable_count = configure_phase2_unfreeze(base_model, unfreeze_top_n=120)
        assert trainable_count > 0
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                assert layer.trainable is False

        sched = build_cosine_warmup_schedule(
            initial_lr=1e-4,
            total_steps=100,
            warmup_steps=10,
            min_lr=1e-6,
        )
        lr_start = float(sched(0))
        lr_peak = float(sched(10))
        lr_mid = float(sched(55))
        lr_end = float(sched(100))

        assert lr_start < lr_peak
        np.testing.assert_allclose(lr_peak, 1e-4, rtol=1e-4)
        assert lr_end < lr_mid < lr_peak
        np.testing.assert_allclose(lr_end, 1e-6, rtol=1e-3)


class TestCalibrationAndOOD:
    """Post-hoc temperature scaling, ECE, and OOD detection."""

    def test_temperature_scaling_lowers_nll_on_overconfident_logits(self) -> None:
        from src.ml.calibration import (
            compute_ece,
            fit_temperature,
            negative_log_likelihood,
            softmax_with_temperature,
        )

        rng = np.random.default_rng(42)
        num_samples, num_classes = 300, 10
        labels = rng.integers(0, num_classes, size=num_samples)
        # Construct strongly overconfident logits with ~25% errors
        logits = rng.normal(0.0, 2.0, size=(num_samples, num_classes))
        for i in range(num_samples):
            if i % 4 != 0:
                logits[i, labels[i]] += 8.0
            else:
                wrong = (labels[i] + 1) % num_classes
                logits[i, wrong] += 8.0

        nll_before = negative_log_likelihood(logits, labels, temperature=1.0)
        best_t = fit_temperature(logits, labels)
        nll_after = negative_log_likelihood(logits, labels, temperature=best_t)

        assert best_t > 1.0
        assert nll_after < nll_before

        ece_before = compute_ece(softmax_with_temperature(logits, 1.0), labels)
        ece_after = compute_ece(softmax_with_temperature(logits, best_t), labels)
        assert ece_after < ece_before

    def test_ood_detector_separates_uniform_noise_from_peaked_predictions(self, tmp_path: Path) -> None:
        from src.ml.calibration import OODDetector

        detector = OODDetector(temperature=1.2, entropy_threshold=0.65, min_confidence=0.45)

        peaked_logits = np.array([[6.0] + [-2.0] * 27], dtype=np.float32)
        uniform_logits = np.zeros((1, 28), dtype=np.float32)

        in_dist = detector.score(peaked_logits)
        ood = detector.score(uniform_logits)

        assert bool(in_dist["is_ood"][0]) is False
        assert bool(ood["is_ood"][0]) is True
        assert float(in_dist["normalized_entropy"][0]) < float(ood["normalized_entropy"][0])

        cfg_path = tmp_path / "model_config.json"
        labels = [f"Class_{i}" for i in range(28)]
        cfg = detector.export_model_config(cfg_path, labels=labels, version="4.0.0", input_size=300)
        loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert loaded == cfg
        assert set(loaded.keys()) == {
            "version",
            "input_size",
            "temperature",
            "ood_entropy_threshold",
            "ood_min_confidence",
            "num_classes",
            "labels",
        }
        assert loaded["num_classes"] == 28


class TestLeakageAuditHardening:
    """P1-4: Active 28-class filter, None on corrupt dHash, cross-class quarantine, clustering."""

    def test_corrupt_image_returns_none_and_does_not_collide(self, tmp_path: Path) -> None:
        from scripts.audit_dataset_leakage import (
            compute_dhash_int,
            scan_dataset_duplicates,
        )

        c1 = tmp_path / "ClassA"
        c2 = tmp_path / "ClassB"
        c1.mkdir()
        c2.mkdir()
        corrupt_a = c1 / "corrupt_a.jpg"
        corrupt_b = c2 / "corrupt_b.jpg"
        corrupt_a.write_bytes(b"not an image")
        corrupt_b.write_bytes(b"also broken")

        assert compute_dhash_int(str(corrupt_a)) is None
        file_hashes, duplicates = scan_dataset_duplicates(str(tmp_path), threshold=3)
        assert len(file_hashes) == 0
        assert len(duplicates) == 0

    def test_active_labels_filter_and_cross_class_quarantine_and_clusters(self, tmp_path: Path) -> None:
        from scripts.audit_dataset_leakage import (
            build_duplicate_clusters,
            partition_duplicate_actions,
            scan_dataset_duplicates,
        )

        (tmp_path / "Ceplok").mkdir()
        (tmp_path / "Yogyakarta_Kawung").mkdir()
        (tmp_path / "Extra_Unused_Folder").mkdir()

        # Cross-class identical pair
        img_cross = _create_rgb_image(tmp_path / "Ceplok" / "dup.png", width=64, height=64, seed=7)
        (tmp_path / "Yogyakarta_Kawung" / "dup.png").write_bytes(img_cross.read_bytes())

        # Intra-class identical pair
        img_intra = _create_rgb_image(tmp_path / "Ceplok" / "intra_1.png", width=64, height=64, seed=9)
        (tmp_path / "Ceplok" / "intra_2.png").write_bytes(img_intra.read_bytes())

        # Unused extra folder image (should be excluded when active_labels is passed)
        _create_rgb_image(tmp_path / "Extra_Unused_Folder" / "extra.png", width=64, height=64, seed=7)

        active = {"Ceplok", "Yogyakarta_Kawung"}
        file_hashes, duplicates = scan_dataset_duplicates(str(tmp_path), threshold=0, active_labels=active)

        assert all(meta["class"] in active for meta in file_hashes.values())
        assert len(file_hashes) == 4

        actions = partition_duplicate_actions(duplicates)
        # Cross-class conflict MUST quarantine BOTH images!
        assert actions["quarantined_cross_class"] == {"Ceplok/dup.png", "Yogyakarta_Kawung/dup.png"}
        # Intra-class duplicate auto-drops exactly 1 of the 2 copies
        assert len(actions["auto_drop_intra_class"]) == 1
        assert actions["auto_drop_intra_class"].issubset({"Ceplok/intra_1.png", "Ceplok/intra_2.png"})

        clusters = build_duplicate_clusters(list(file_hashes.keys()), duplicates)
        assert clusters["Ceplok/dup.png"] == clusters["Yogyakarta_Kawung/dup.png"]
        assert clusters["Ceplok/intra_1.png"] == clusters["Ceplok/intra_2.png"]
        assert clusters["Ceplok/dup.png"] != clusters["Ceplok/intra_1.png"]


class TestColabTrainScriptCLI:
    """Verify headless Colab v4 training script compiles and exposes CLI help."""

    def test_train_v4_colab_help_exits_zero(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "scripts" / "train_v4_colab.py"
        assert script_path.exists(), f"Missing script: {script_path}"

        res = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0
        for flag in ("--dataset-dir", "--labels-file", "--output-dir", "--backbone", "--img-size", "--folds", "--smoke-test", "--quantize"):
            assert flag in res.stdout

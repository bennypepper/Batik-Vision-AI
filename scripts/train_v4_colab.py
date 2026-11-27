"""Headless Google Colab v4 training pipeline for Indonesian Batik Motifs Classification.

Designed for one-shot headless execution via:
    colab run --gpu L4 --timeout 0 scripts/train_v4_colab.py \
        --dataset-dir /content/dataset \
        --labels-file models/labels.txt \
        --output-dir /content/v4_out \
        --backbone efficientnetv2s \
        --img-size 300 \
        --folds 5 \
        --quantize fp16
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repository root is importable when invoked as a standalone script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for headless v4 training."""
    parser = argparse.ArgumentParser(
        description="Train, calibrate, evaluate, and export Batik Vision AI v4 model.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root directory containing per-class image folders.",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default=str(REPO_ROOT / "models" / "labels.txt"),
        help="Path to active 28-class labels.txt or labels.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/v4_train",
        help="Directory to write checkpoints, ONNX model, model_config.json, and reports.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["efficientnetv2s", "convnext_tiny", "dinov2_vits14"],
        default="efficientnetv2s",
        help="Backbone architecture to train.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        choices=[224, 300, 384],
        default=300,
        help="Square input image resolution.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of StratifiedGroupKFold cross-validation folds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=15,
        help="Max epochs for frozen-backbone head warmup.",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=35,
        help="Max epochs for Phase 2 fine-tuning.",
    )
    parser.add_argument(
        "--unfreeze-top-n",
        type=int,
        default=120,
        help="Number of top backbone layers to unfreeze in Phase 2 (BatchNorm stays frozen).",
    )
    parser.add_argument(
        "--loss-mode",
        type=str,
        choices=["cce_smooth", "focal"],
        default="cce_smooth",
        help="Classification loss mode.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing factor.",
    )
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=3.0,
        help="Upper cap for Effective Number of Samples class weights.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=int,
        default=3,
        help="dHash Hamming distance threshold for clustering/deduplication.",
    )
    parser.add_argument(
        "--use-dinov2-clustering",
        action="store_true",
        help="Optionally augment dHash duplicate clusters with DINOv2 cosine similarity (>0.96).",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["fp16", "int8", "none"],
        default="fp16",
        help="ONNX export quantization mode.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 1-epoch, 2-fold verification pass.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed.",
    )
    return parser.parse_args(argv)


def build_backbone_model(
    backbone: str,
    num_classes: int,
    img_size: int,
    weights: str | None = "imagenet",
):
    """Instantiate the requested Keras backbone with raw [0, 255] float32 input parity."""
    import tensorflow as tf

    from src.ml.pipeline import create_v3_backbone_model

    if backbone == "efficientnetv2s":
        return create_v3_backbone_model(num_classes=num_classes, img_size=img_size, weights=weights)

    if backbone == "convnext_tiny":
        input_shape = (img_size, img_size, 3)
        inputs = tf.keras.Input(shape=input_shape, name="input_image", dtype=tf.float32)
        base_model = tf.keras.applications.ConvNeXtTiny(
            include_top=False,
            weights=weights,
            include_preprocessing=True,
            input_shape=input_shape,
        )
        features = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(features)
        x = tf.keras.layers.LayerNormalization(name="head_ln")(x)
        x = tf.keras.layers.Dropout(0.4, name="head_dropout_1")(x)
        x = tf.keras.layers.Dense(256, activation="gelu", name="head_dense")(x)
        x = tf.keras.layers.Dropout(0.25, name="head_dropout_2")(x)
        outputs = tf.keras.layers.Dense(
            num_classes,
            activation="softmax",
            dtype="float32",
            name="predictions",
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="batik_convnext_tiny_v4")
        return base_model, model

    raise NotImplementedError(
        "Backbone 'dinov2_vits14' requires PyTorch/timm export; use 'efficientnetv2s' or 'convnext_tiny' "
        "for the Keras/TF pipeline, or run the dedicated PyTorch DINOv2 trainer."
    )


def _optional_dinov2_edges(
    file_paths: list[str],
    cosine_threshold: float = 0.96,
) -> list[dict[str, Any]]:
    """Optionally compute high-similarity visual pairs using DINOv2 embeddings if torch/timm exist."""
    try:
        import timm
        import torch
        from PIL import Image
        from torchvision import transforms
    except ImportError:
        print("NOTICE: torch/timm not installed; skipping optional DINOv2 cosine clustering.", flush=True)
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0).to(device)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    embeddings = []
    with torch.no_grad():
        for fp in file_paths:
            with Image.open(fp) as im:
                tensor = preprocess(im.convert("RGB")).unsqueeze(0).to(device)
                feat = model(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                embeddings.append(feat.cpu().numpy()[0])

    emb_mat = np.stack(embeddings, axis=0)
    sim_mat = emb_mat @ emb_mat.T
    edges: list[dict[str, Any]] = []
    n = len(file_paths)
    for i in range(n):
        for j in range(i + 1, n):
            if float(sim_mat[i, j]) >= cosine_threshold:
                edges.append({"file_a": file_paths[i], "file_b": file_paths[j]})
    print(f"DINOv2 clustering found {len(edges)} pairs with cosine >= {cosine_threshold}.", flush=True)
    return edges


def run_training(args: argparse.Namespace) -> None:
    """Execute the full leak-free v4 training, calibration, evaluation, and export pipeline."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    from scripts.audit_dataset_leakage import (
        build_duplicate_clusters,
        load_active_labels,
        partition_duplicate_actions,
        scan_dataset_duplicates,
    )
    from src.ml.augmentations import (
        apply_augmentation_to_float32,
        build_eval_augmentation,
        build_train_augmentation,
        is_orientation_sensitive,
        load_image_uint8,
    )
    from src.ml.calibration import (
        OODDetector,
        compute_ece,
        fit_temperature,
        softmax_with_temperature,
    )
    from src.ml.losses import build_classification_loss
    from src.ml.pipeline import (
        build_cosine_warmup_schedule,
        compute_effective_class_weights,
        configure_phase2_unfreeze,
    )

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(f"GPU(s) detected: {len(gpus)} (mixed_float16 enabled)", flush=True)
    else:
        print("Running on CPU (no GPU detected).", flush=True)

    # 1. Strict active-class dataset audit, quarantine, and clustering
    active_labels = load_active_labels(args.labels_file)
    class_names = sorted(active_labels)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)

    print(f"Auditing dataset at {args.dataset_dir} against {num_classes} active classes...", flush=True)
    file_hashes, duplicates = scan_dataset_duplicates(
        args.dataset_dir,
        threshold=args.dedup_threshold,
        active_labels=active_labels,
    )
    actions = partition_duplicate_actions(duplicates)
    excluded = actions["quarantined_cross_class"] | actions["auto_drop_intra_class"]
    print(
        f"Valid hashed: {len(file_hashes)} | Intra-class drops: {len(actions['auto_drop_intra_class'])} "
        f"| Quarantined cross-class: {len(actions['quarantined_cross_class'])}",
        flush=True,
    )

    kept_rel_paths = [rp for rp in sorted(file_hashes.keys()) if rp not in excluded]
    cluster_edges = list(duplicates)
    if args.use_dinov2_clustering and kept_rel_paths:
        abs_list = [file_hashes[rp]["abs_path"] for rp in kept_rel_paths]
        abs_to_rel = {file_hashes[rp]["abs_path"]: rp for rp in kept_rel_paths}
        for edge in _optional_dinov2_edges(abs_list):
            cluster_edges.append({
                "file_a": abs_to_rel[edge["file_a"]],
                "file_b": abs_to_rel[edge["file_b"]],
            })

    clusters = build_duplicate_clusters(kept_rel_paths, cluster_edges)

    filepaths = np.array([file_hashes[rp]["abs_path"] for rp in kept_rel_paths])
    labels = np.array([class_to_idx[file_hashes[rp]["class"]] for rp in kept_rel_paths], dtype=np.int32)
    groups = np.array([clusters[rp] for rp in kept_rel_paths], dtype=np.int32)

    # 2. Stratified holdout + StratifiedGroupKFold
    X_trainval, X_test, y_trainval, y_test, g_trainval, _ = train_test_split(
        filepaths,
        labels,
        groups,
        test_size=0.15,
        random_state=args.seed,
        stratify=labels,
    )

    class_counts_all = {c: int(np.sum(labels == idx)) for idx, c in enumerate(class_names)}
    tier_map = {
        idx: ("tier1" if class_counts_all[name] <= 45 else ("tier2" if class_counts_all[name] <= 90 else "tier3"))
        for idx, name in enumerate(class_names)
    }
    flip_map = {idx: not is_orientation_sensitive(name) for idx, name in enumerate(class_names)}

    train_pipelines = {
        (tier, allow_flip): build_train_augmentation(
            img_size=args.img_size,
            tier=tier,
            allow_horizontal_flip=allow_flip,
        )
        for tier in ("tier1", "tier2", "tier3")
        for allow_flip in (True, False)
    }
    eval_pipeline = build_eval_augmentation(img_size=args.img_size)

    def _aug_train(fp_bytes, lbl):
        img_u8 = load_image_uint8(fp_bytes.numpy().decode("utf-8"), fallback_size=args.img_size)
        lbl_int = int(lbl.numpy())
        tfm = train_pipelines[(tier_map[lbl_int], flip_map[lbl_int])]
        return apply_augmentation_to_float32(img_u8, tfm), lbl_int

    def _aug_eval(fp_bytes, lbl):
        img_u8 = load_image_uint8(fp_bytes.numpy().decode("utf-8"), fallback_size=args.img_size)
        return apply_augmentation_to_float32(img_u8, eval_pipeline), int(lbl.numpy())

    def _make_ds(fps, lbls, fn, shuffle: bool = False):
        def _tf_map(fp, lb):
            img, out_lb = tf.py_function(func=fn, inp=[fp, lb], Tout=[tf.float32, tf.int32])
            img.set_shape([args.img_size, args.img_size, 3])
            out_lb.set_shape([])
            return img, tf.one_hot(out_lb, depth=num_classes)

        ds = tf.data.Dataset.from_tensor_slices((fps, lbls))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(fps), seed=args.seed, reshuffle_each_iteration=True)
        return ds.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    n_folds = 2 if args.smoke_test else args.folds
    p1_epochs = 1 if args.smoke_test else args.phase1_epochs
    p2_epochs = 1 if args.smoke_test else args.phase2_epochs

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    best_model_path = output_dir / "best_batik_model_v4.keras"
    best_overall_val_acc = -1.0
    best_val_ds_info: tuple[np.ndarray, np.ndarray] | None = None

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_trainval, y_trainval, groups=g_trainval)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===", flush=True)
        X_tr, y_tr = X_trainval[tr_idx], y_trainval[tr_idx]
        X_va, y_va = X_trainval[va_idx], y_trainval[va_idx]

        tr_counts = [int(np.sum(y_tr == c)) for c in range(num_classes)]
        weights_arr = compute_effective_class_weights(tr_counts, beta=0.999, max_weight=args.max_class_weight)
        fold_class_weights = dict(enumerate(weights_arr))

        train_ds = _make_ds(X_tr, y_tr, _aug_train, shuffle=True)
        val_ds = _make_ds(X_va, y_va, _aug_eval, shuffle=False)

        base_model, model = build_backbone_model(
            backbone=args.backbone,
            num_classes=num_classes,
            img_size=args.img_size,
            weights=None if args.smoke_test else "imagenet",
        )
        loss_fn = build_classification_loss(
            mode=args.loss_mode,
            label_smoothing=args.label_smoothing,
        )

        # Phase 1: Frozen backbone
        base_model.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss_fn,
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=p1_epochs,
            class_weight=fold_class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max"),
            ],
            verbose=1,
        )

        # Phase 2: Unfreeze top N layers (BatchNorm frozen) + Cosine Warmup
        trainable_layers = configure_phase2_unfreeze(base_model, unfreeze_top_n=args.unfreeze_top_n)
        steps_per_epoch = max(1, math.ceil(len(X_tr) / args.batch_size))
        total_steps = steps_per_epoch * p2_epochs
        warmup_steps = max(1, int(0.1 * total_steps))
        p2_schedule = build_cosine_warmup_schedule(
            initial_lr=5e-5,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=1e-7,
        )

        fold_ckpt = output_dir / f"fold_{fold + 1}_best.keras"
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=p2_schedule),
            loss=loss_fn,
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
        )
        print(f"Phase 2 fine-tuning ({trainable_layers} trainable backbone layers)...", flush=True)
        hist_p2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=p2_epochs,
            class_weight=fold_class_weights,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(fold_ckpt),
                    monitor="val_accuracy",
                    save_best_only=True,
                    mode="max",
                ),
                tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, mode="max"),
            ],
            verbose=1,
        )

        fold_best = max(hist_p2.history["val_accuracy"])
        if fold_best > best_overall_val_acc and fold_ckpt.exists():
            best_overall_val_acc = fold_best
            shutil.copy2(fold_ckpt, best_model_path)
            best_val_ds_info = (X_va, y_va)
            print(f"Copied new best checkpoint (val_acc={fold_best:.4f}) -> {best_model_path}", flush=True)

        del model, base_model, train_ds, val_ds
        tf.keras.backend.clear_session()
        gc.collect()

    # 3. Temperature Scaling + OOD Calibration on best fold validation split
    print("\nRunning post-hoc Temperature Scaling & OOD calibration...", flush=True)
    best_model = tf.keras.models.load_model(best_model_path, compile=False)
    pre_softmax_tensor = best_model.get_layer("predictions").input
    logits_head = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        weights=best_model.get_layer("predictions").get_weights(),
        name="logits_head",
    )(pre_softmax_tensor)
    logits_model = tf.keras.Model(inputs=best_model.input, outputs=logits_head)

    assert best_val_ds_info is not None
    val_ds_cal = _make_ds(best_val_ds_info[0], best_val_ds_info[1], _aug_eval, shuffle=False)
    val_logits = logits_model.predict(val_ds_cal, verbose=0)
    best_temp = fit_temperature(val_logits, best_val_ds_info[1])
    ece_before = compute_ece(softmax_with_temperature(val_logits, 1.0), best_val_ds_info[1])
    ece_after = compute_ece(softmax_with_temperature(val_logits, best_temp), best_val_ds_info[1])
    print(f"Optimal Temperature T = {best_temp:.4f} | Val ECE: {ece_before:.4f} -> {ece_after:.4f}", flush=True)

    detector = OODDetector(temperature=best_temp, entropy_threshold=0.65, min_confidence=0.40)
    detector.export_model_config(
        output_dir / "model_config.json",
        labels=class_names,
        version="4.0.0",
        input_size=args.img_size,
    )

    # 4. Holdout Test Evaluation + Classification Report + Confusion Matrix
    test_ds = _make_ds(X_test, y_test, _aug_eval, shuffle=False)
    test_logits = logits_model.predict(test_ds, verbose=0)
    test_probs = softmax_with_temperature(test_logits, temperature=best_temp)
    y_pred = np.argmax(test_probs, axis=1)

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0)
    (output_dir / "v4_classification_report.txt").write_text(report, encoding="utf-8")
    print("\nHoldout Classification Report:\n" + report, flush=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Batik Vision AI v4 Confusion Matrix ({args.backbone}, {args.img_size}px)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "v4_confusion_matrix.png", dpi=150)
    plt.close()

    # 5. Export ONNX (opset 17 + requested quantization)
    onnx_out = output_dir / "batik_model_v4.onnx"
    convert_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "convert_keras_to_onnx.py"),
        "--input",
        str(best_model_path),
        "--output",
        str(onnx_out),
        "--img-size",
        str(args.img_size),
        "--quantize",
        args.quantize,
    ]
    subprocess.run(convert_cmd, check=True)

    summary = {
        "backbone": args.backbone,
        "img_size": args.img_size,
        "best_val_accuracy": float(best_overall_val_acc),
        "temperature": float(best_temp),
        "val_ece_before": float(ece_before),
        "val_ece_after": float(ece_after),
        "test_top1_accuracy": float(np.mean(y_pred == y_test)),
        "quantize": args.quantize,
    }
    (output_dir / "v4_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nTraining complete. Summary saved to {output_dir / 'v4_run_summary.json'}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()

"""Cell 5: Stratified 5-Fold cross-validation training loop with two-phase fine-tuning."""

import gc
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import TopKCategoricalAccuracy

fold_records = []
best_overall_val_acc = 0.0
best_overall_model_path = MODEL_PATH

for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
    print(f"\n{'='*65}")
    print(f"  STARTING STRATIFIED FOLD {fold + 1} / {N_FOLDS}")
    print(f"{'='*65}")

    X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]

    # Compute effective sample weights for current fold
    tr_counts = [np.sum(y_tr == c) for c in range(NUM_CLASSES)]
    class_weights_arr = compute_effective_sample_weights(tr_counts, beta=EFFECTIVE_BETA)
    fold_class_weights = dict(enumerate(class_weights_arr))

    train_ds_fold = build_dataset(X_tr, y_tr, tf_augment_train, BATCH_SIZE, shuffle=True)
    val_ds_fold = build_dataset(X_val, y_val, tf_augment_eval, BATCH_SIZE, shuffle=False)

    base_model, model = create_v3_backbone_model(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE, weights="imagenet")

    # Phase 1: Train classification head with frozen backbone
    base_model.trainable = False

    loss_fn = CategoricalFocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)

    model.compile(
        optimizer=Adam(learning_rate=PHASE1_LR),
        loss=loss_fn,
        metrics=["accuracy", TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )

    print(f"  [Fold {fold + 1}] Phase 1: Training classification head ({PHASE1_EPOCHS} epochs max)...")
    history_p1 = model.fit(
        train_ds_fold,
        validation_data=val_ds_fold,
        epochs=PHASE1_EPOCHS,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=PHASE1_PATIENCE, restore_best_weights=True, verbose=0, mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=0),
        ],
        class_weight=fold_class_weights,
        verbose=1,
    )

    # Phase 2: Unfreeze top N layers; keep all BatchNorm layers frozen
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    fold_ckpt_path = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}_best.keras")

    model.compile(
        optimizer=Adam(learning_rate=PHASE2_LR),
        loss=loss_fn,
        metrics=["accuracy", TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )

    print(f"  [Fold {fold + 1}] Phase 2: Fine-tuning last {UNFREEZE_LAYERS} layers ({PHASE2_EPOCHS} epochs max)...")
    history_p2 = model.fit(
        train_ds_fold,
        validation_data=val_ds_fold,
        epochs=PHASE2_EPOCHS,
        callbacks=[
            ModelCheckpoint(filepath=fold_ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=0),
            EarlyStopping(monitor="val_accuracy", patience=PHASE2_PATIENCE, restore_best_weights=True, verbose=0, mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=0),
        ],
        class_weight=fold_class_weights,
        verbose=1,
    )

    best_val_acc = max(history_p2.history["val_accuracy"])
    best_val_top3 = max(history_p2.history["val_top3_accuracy"])

    if best_val_acc > best_overall_val_acc:
        best_overall_val_acc = best_val_acc
        model.save(best_overall_model_path)
        print(f"  --> New overall best model saved: val_acc = {best_val_acc:.4f}")

    fold_records.append({
        "fold": fold + 1,
        "val_accuracy": float(best_val_acc),
        "val_top3_accuracy": float(best_val_top3),
    })
    print(f"  Fold {fold + 1} Results -> Val Acc: {best_val_acc:.4f} | Val Top-3: {best_val_top3:.4f}")

    del model, base_model, train_ds_fold, val_ds_fold
    tf.keras.backend.clear_session()
    gc.collect()

print("\n" + "=" * 65)
print("  5-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 65)
for rec in fold_records:
    print(f"  Fold {rec['fold']} -> Top-1: {rec['val_accuracy']:.4f} | Top-3: {rec['val_top3_accuracy']:.4f}")
mean_acc = np.mean([r["val_accuracy"] for r in fold_records])
print(f"\nMean Validation Accuracy across {N_FOLDS} folds: {mean_acc:.4f}")

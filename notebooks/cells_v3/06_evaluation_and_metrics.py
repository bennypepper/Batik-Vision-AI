"""Cell 6: Test holdout evaluation, Test-Time Augmentation (TTA), and comprehensive metrics."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

print("Loading best model for test holdout evaluation...")
best_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"CategoricalFocalLoss": CategoricalFocalLoss},
)

# Standard evaluation on clean test set
print("Evaluating on clean test holdout...")
clean_eval = best_model.evaluate(test_ds, verbose=1)
print(f"Clean Test Holdout Loss    : {clean_eval[0]:.4f}")
print(f"Clean Test Holdout Accuracy: {clean_eval[1]:.4f}")

# Test-Time Augmentation (TTA)
print(f"\nExecuting {TTA_STEPS}-step Test-Time Augmentation (TTA)...")
tta_predictions = []

# Step 0: Clean pass
clean_preds = best_model.predict(test_ds, verbose=0)
tta_predictions.append(clean_preds)

# Steps 1..TTA_STEPS: Lightly augmented passes
for step in range(TTA_STEPS):
    tta_ds = build_dataset(X_test, y_test, tf_augment_tta, BATCH_SIZE, shuffle=False)
    preds = best_model.predict(tta_ds, verbose=0)
    tta_predictions.append(preds)

# Ensemble average
ensemble_probs = np.mean(tta_predictions, axis=0)
y_pred_classes = np.argmax(ensemble_probs, axis=1)

# Overall Accuracy Metrics
top1_acc = np.mean(y_pred_classes == y_test)
top3_hits = [y_test[i] in np.argsort(ensemble_probs[i])[-3:] for i in range(len(y_test))]
top3_acc = np.mean(top3_hits)

print("\n" + "=" * 65)
print("  FINAL TEST HOLDOUT METRICS (WITH TTA)")
print("=" * 65)
print(f"  Top-1 Accuracy : {top1_acc * 100:.2f}%")
print(f"  Top-3 Accuracy : {top3_acc * 100:.2f}%")

# Classification Report
target_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
clf_report = classification_report(y_test, y_pred_classes, target_names=target_names, digits=4)
print("\nPer-Class Classification Report:")
print(clf_report)

# Save report to disk
report_path = os.path.join(OUTPUT_DIR, "v3_classification_report.txt")
with open(report_path, "w") as f:
    f.write(clf_report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(18, 16))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.title("Confusion Matrix (V3 Test Holdout with TTA)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, "v3_confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved confusion matrix visualization to: {cm_path}")

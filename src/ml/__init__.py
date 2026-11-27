"""Machine learning pipeline, augmentations, calibration, and losses for Batik Motifs v3/v4."""

from src.ml.augmentations import (
    ORIENTATION_SENSITIVE_CLASSES,
    apply_augmentation_to_float32,
    augment_image_file,
    build_eval_augmentation,
    build_train_augmentation,
    is_orientation_sensitive,
    load_image_uint8,
)
from src.ml.calibration import (
    OODDetector,
    compute_ece,
    energy_score,
    fit_temperature,
    negative_log_likelihood,
    normalized_entropy,
    softmax_with_temperature,
)
from src.ml.losses import CategoricalFocalLoss, build_classification_loss
from src.ml.pipeline import (
    CosineWarmupSchedule,
    build_cosine_warmup_schedule,
    compute_effective_class_weights,
    compute_effective_sample_weights,
    configure_phase2_unfreeze,
    create_v3_backbone_model,
)

__all__ = [
    "ORIENTATION_SENSITIVE_CLASSES",
    "CategoricalFocalLoss",
    "CosineWarmupSchedule",
    "OODDetector",
    "apply_augmentation_to_float32",
    "augment_image_file",
    "build_classification_loss",
    "build_cosine_warmup_schedule",
    "build_eval_augmentation",
    "build_train_augmentation",
    "compute_ece",
    "compute_effective_class_weights",
    "compute_effective_sample_weights",
    "configure_phase2_unfreeze",
    "create_v3_backbone_model",
    "energy_score",
    "fit_temperature",
    "is_orientation_sensitive",
    "load_image_uint8",
    "negative_log_likelihood",
    "normalized_entropy",
    "softmax_with_temperature",
]

"""Cell 3: Group-stratified holdout, domain-safe uint8 Albumentations, and tf.data loader."""

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# Fixed 15% test holdout (never touched during K-Fold training or validation)
X_trainval, X_test, y_trainval, y_test, g_trainval, g_test = train_test_split(
    df["filepath"].values,
    df["label_encoded"].values,
    df["cluster_id"].values,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df["label_encoded"].values,
)

print(f"Fixed test holdout pool : {len(X_test)} images")
print(f"Train/Validation pool   : {len(X_trainval)} images")

skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Orientation-sensitive motifs (Parang diagonal polarity, Lontara script) must NOT be flipped
ORIENTATION_SENSITIVE_CLASSES = {
    "Solo_Parang",
    "Yogyakarta_Parang",
    "Sulawesi_Selatan_Lontara",
}


def build_tier_pipeline(tier: str, allow_hflip: bool = True) -> A.Compose:
    """Domain-safe uint8 Albumentations pipeline (no GridDistortion/ElasticTransform, bounded HSV)."""
    ops = [A.HorizontalFlip(p=0.5)] if allow_hflip else []
    if tier == "tier1":
        ops.extend([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.5),
            A.RandomResizedCrop(size=IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.25),
            A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.25),
            A.Sharpen(alpha=(0.1, 0.25), lightness=(0.85, 1.15), p=0.2),
        ])
    elif tier == "tier2":
        ops.extend([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.4),
            A.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.4),
            A.GaussianBlur(blur_limit=(3, 3), p=0.15),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        ])
    else:
        ops.extend([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=12, val_shift_limit=10, p=0.3),
        ])
    ops.append(A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH))
    return A.Compose(ops)


AUGMENT_PIPELINES = {
    (tier, allow_flip): build_tier_pipeline(tier, allow_hflip=allow_flip)
    for tier in ("tier1", "tier2", "tier3")
    for allow_flip in (True, False)
}

AUGMENT_VALIDATION = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

AUGMENT_TTA = build_tier_pipeline("tier3", allow_hflip=False)


def get_tier_for_class(class_name: str) -> str:
    count = df[df["label"] == class_name].shape[0]
    if count <= TIER1_THRESHOLD:
        return "tier1"
    if count <= TIER2_THRESHOLD:
        return "tier2"
    return "tier3"


CLASS_TIER_MAP = {i: get_tier_for_class(CLASS_NAMES[i]) for i in range(NUM_CLASSES)}
CLASS_ALLOW_FLIP_MAP = {i: CLASS_NAMES[i] not in ORIENTATION_SENSITIVE_CLASSES for i in range(NUM_CLASSES)}


def load_image_uint8(filepath: str) -> np.ndarray:
    """Load raw native-resolution RGB uint8 image (NEVER pre-resize or cast to float32)."""
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def augment_train(filepath: bytes, label: int) -> tuple:
    img_uint8 = load_image_uint8(filepath.numpy().decode("utf-8"))
    label_int = int(label.numpy())
    tier = CLASS_TIER_MAP.get(label_int, "tier2")
    allow_flip = CLASS_ALLOW_FLIP_MAP.get(label_int, True)
    img_aug = AUGMENT_PIPELINES[(tier, allow_flip)](image=img_uint8)["image"]
    return img_aug.astype(np.float32), label_int


def augment_eval(filepath: bytes, label: int) -> tuple:
    img_uint8 = load_image_uint8(filepath.numpy().decode("utf-8"))
    img_aug = AUGMENT_VALIDATION(image=img_uint8)["image"]
    return img_aug.astype(np.float32), int(label.numpy())


def augment_tta(filepath: bytes, label: int) -> tuple:
    img_uint8 = load_image_uint8(filepath.numpy().decode("utf-8"))
    img_aug = AUGMENT_TTA(image=img_uint8)["image"]
    return img_aug.astype(np.float32), int(label.numpy())


def _wrap(aug_fn):
    """Wrap Python numpy augmentation function for tf.data graph execution.

    Note that preprocess_input is intentionally OMITTED here: the model's
    internal Rescaling(1/127.5, offset=-1.0) layer processes raw [0, 255] float32.
    """
    def tf_fn(filepath, label):
        img, lbl = tf.py_function(func=aug_fn, inp=[filepath, label], Tout=[tf.float32, tf.int32])
        img.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        lbl.set_shape([])
        return img, tf.one_hot(lbl, depth=NUM_CLASSES)

    return tf_fn


tf_augment_train = _wrap(augment_train)
tf_augment_eval = _wrap(augment_eval)
tf_augment_tta = _wrap(augment_tta)

AUTOTUNE = tf.data.AUTOTUNE


def build_dataset(filepaths, labels, augment_fn, batch_size, shuffle=False, cache=False):
    """Construct optimized tf.data.Dataset pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    return ds.batch(batch_size).prefetch(AUTOTUNE)


test_ds = build_dataset(X_test, y_test, tf_augment_eval, BATCH_SIZE, shuffle=False, cache=True)

print("Augmentation tier summary:")
for tier_name, tier_key in [("Tier 1 (Aggressive)", "tier1"), ("Tier 2 (Moderate)", "tier2"), ("Tier 3 (Light)", "tier3")]:
    classes_in_tier = [CLASS_NAMES[i] for i, t in CLASS_TIER_MAP.items() if t == tier_key]
    print(f"  {tier_name}: {len(classes_in_tier)} classes")

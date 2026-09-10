"""Cell 3: Stratified train/test holdout, Albumentations tier pipelines, and tf.data loader."""

import cv2
import numpy as np
import albumentations as A
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold

# Fixed 15% test holdout (never touched during K-Fold training or validation)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    df["filepath"].values,
    df["label_encoded"].values,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df["label_encoded"].values,
)

print(f"Fixed test holdout pool : {len(X_test)} images")
print(f"Train/Validation pool   : {len(X_trainval)} images")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Tier-based Albumentations pipelines
AUGMENT_TIER1 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.6),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
    A.ElasticTransform(alpha=80, sigma=8, p=0.3),
    A.RandomResizedCrop(size=IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

AUGMENT_TIER2 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.4),
    A.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.4),
    A.GaussianBlur(blur_limit=(3, 3), p=0.2),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

AUGMENT_TIER3 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

AUGMENT_VALIDATION = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

AUGMENT_TTA = AUGMENT_TIER3


def get_tier_for_class(class_name: str) -> str:
    count = df[df["label"] == class_name].shape[0]
    if count <= TIER1_THRESHOLD:
        return "tier1"
    elif count <= TIER2_THRESHOLD:
        return "tier2"
    return "tier3"


CLASS_TIER_MAP = {i: get_tier_for_class(CLASS_NAMES[i]) for i in range(NUM_CLASSES)}


def load_image(filepath: str) -> np.ndarray:
    """Loads image from disk in RGB format. Returns raw float32 pixels in [0, 255]."""
    img = cv2.imread(filepath)
    if img is None:
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32)


def augment_train(filepath: bytes, label: int) -> tuple:
    img = load_image(filepath.numpy().decode("utf-8"))
    label_int = int(label.numpy())
    tier = CLASS_TIER_MAP.get(label_int, "tier2")
    if tier == "tier1":
        img = AUGMENT_TIER1(image=img)["image"]
    elif tier == "tier2":
        img = AUGMENT_TIER2(image=img)["image"]
    else:
        img = AUGMENT_TIER3(image=img)["image"]
    return img.astype(np.float32), label_int


def augment_eval(filepath: bytes, label: int) -> tuple:
    img = load_image(filepath.numpy().decode("utf-8"))
    img = AUGMENT_VALIDATION(image=img)["image"]
    return img.astype(np.float32), int(label.numpy())


def augment_tta(filepath: bytes, label: int) -> tuple:
    img = load_image(filepath.numpy().decode("utf-8"))
    img = AUGMENT_TTA(image=img)["image"]
    return img.astype(np.float32), int(label.numpy())


def _wrap(aug_fn):
    """Wraps Python numpy augmentation function for tf.data graph execution.

    CRITICAL FIX: Note that preprocess_input is intentionally OMITTED here.
    The model's internal Rescaling layer directly processes raw [0, 255] float32 pixels.
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
    """Constructs optimized tf.data.Dataset pipeline."""
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

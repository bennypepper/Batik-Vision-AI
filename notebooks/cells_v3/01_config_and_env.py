"""Cell 1: Environment detection, deterministic seeding, and hyperparameters."""

import os
import platform
import random
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def auto_detect_dataset_dir() -> tuple[str, str]:
    """Detect dataset and output directories from env vars, Kaggle, Colab, or local workspace."""
    env_dataset = os.environ.get("BATIK_DATASET_DIR")
    env_output = os.environ.get("BATIK_OUTPUT_DIR", "./outputs/v3_train")
    if env_dataset and Path(env_dataset).exists():
        return env_dataset, env_output

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.rglob("*"):
            if p.is_dir() and (p / "Bali_Barong").exists():
                return str(p), "/kaggle/working"

    local_paths = [
        Path("/content/dataset"),
        Path("data/dataset"),
        Path("./dataset"),
    ]
    for lp in local_paths:
        if lp.exists() and (lp / "Bali_Barong").exists():
            return str(lp), env_output

    return "./dataset", env_output


DATASET_DIR, OUTPUT_DIR = auto_detect_dataset_dir()
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_batik_model_v3.keras")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "batik_model_v3.tflite")
ONNX_PATH = os.path.join(OUTPUT_DIR, "batik_model_v3.onnx")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.txt")
AUDIT_CSV_PATH = os.path.join(OUTPUT_DIR, "v3_dataset_audit.csv")
ACTIVE_LABELS_FILE = os.environ.get("BATIK_LABELS_FILE", "models/labels.txt")

# Image input dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Training hyperparameters
BATCH_SIZE = 32
N_FOLDS = 5
TEST_SIZE = 0.15
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 40
PHASE1_LR = 1e-3
PHASE2_LR = 5e-5
PHASE1_PATIENCE = 5
PHASE2_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.3
MIN_LR = 1e-7

# Model architecture & loss hyperparameters
UNFREEZE_LAYERS = 120
DROPOUT_RATE_1 = 0.5
DROPOUT_RATE_2 = 0.3
DENSE_UNITS = 256
LOSS_MODE = "cce_smooth"
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05
EFFECTIVE_BETA = 0.999
MAX_CLASS_WEIGHT = 3.0
TTA_STEPS = 5

TIER1_THRESHOLD = 45
TIER2_THRESHOLD = 90

# GPU memory growth and mixed precision configuration
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU devices detected: {len(gpus)}")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"Mixed precision policy set to: {tf.keras.mixed_precision.global_policy().name}")
else:
    print("WARNING: No GPU detected. Execution will proceed on CPU.")

print("\nRuntime Configuration:")
print(f"  Platform         : {platform.platform()}")
print(f"  Python           : {platform.python_version()}")
print(f"  TensorFlow       : {tf.__version__}")
print("  Backbone         : EfficientNetV2S (v3/v4 hardened pipeline)")
print(f"  Dataset directory: {DATASET_DIR}")
print(f"  Output directory : {OUTPUT_DIR}")
print(f"  Image dimensions : {IMG_SIZE}")
print(f"  Batch size       : {BATCH_SIZE}")
print(f"  Stratified Folds : {N_FOLDS}")
print(f"  Loss mode        : {LOSS_MODE} (label_smoothing={LABEL_SMOOTHING})")
print(f"  Unfreeze top N   : {UNFREEZE_LAYERS}")

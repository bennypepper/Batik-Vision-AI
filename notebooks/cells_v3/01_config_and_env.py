"""Cell 1: Environment detection, deterministic seeding, and hyperparameters."""

import os
import random
import platform
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

# Detect runtime environment paths
KAGGLE_DATASET = Path("/kaggle/input/datasets/fisheightcharacter/indonesian-batik-dataset-enhanced-and-cleaned/dataset")
LOCAL_DATASET = Path(r"D:\Personal Projects\Computer Vision-ML Projects\Batik Motifs Detector v2\Batik_dataset\dataset")
REPO_DATASET = Path("data/dataset")

if KAGGLE_DATASET.exists():
    DATASET_DIR = str(KAGGLE_DATASET)
    OUTPUT_DIR = "/kaggle/working"
elif LOCAL_DATASET.exists():
    DATASET_DIR = str(LOCAL_DATASET)
    OUTPUT_DIR = "./outputs/v3_train"
elif REPO_DATASET.exists():
    DATASET_DIR = str(REPO_DATASET)
    OUTPUT_DIR = "./outputs/v3_train"
else:
    DATASET_DIR = "./dataset"
    OUTPUT_DIR = "./outputs/v3_train"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_batik_model_v3.keras")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "batik_model_v3.tflite")
ONNX_PATH = os.path.join(OUTPUT_DIR, "batik_model_v3.onnx")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.txt")
AUDIT_CSV_PATH = os.path.join(OUTPUT_DIR, "v3_dataset_audit.csv")

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
PHASE2_LR = 1e-5
PHASE1_PATIENCE = 5
PHASE2_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.3
MIN_LR = 1e-7

# Model architecture hyperparameters
UNFREEZE_LAYERS = 40
DROPOUT_RATE_1 = 0.5
DROPOUT_RATE_2 = 0.3
DENSE_UNITS = 256
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1
EFFECTIVE_BETA = 0.999
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
print(f"  Backbone         : EfficientNetV2S (v3 fixed pipeline)")
print(f"  Dataset directory: {DATASET_DIR}")
print(f"  Output directory : {OUTPUT_DIR}")
print(f"  Image dimensions : {IMG_SIZE}")
print(f"  Batch size       : {BATCH_SIZE}")
print(f"  Stratified Folds : {N_FOLDS}")
print(f"  Focal loss gamma : {FOCAL_GAMMA}")
print(f"  Label smoothing  : {LABEL_SMOOTHING}")

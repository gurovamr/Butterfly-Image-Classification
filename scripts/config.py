"""Configuration constants and hyperparameters for butterfly classification."""

from __future__ import annotations
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
# Data download
ZENODO_URL = "https://zenodo.org/records/7559420/files/leedsbutterfly_dataset_v1.1.zip?download=1"

# Image preprocessing
IMAGE_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)

# Model architecture
NUM_CLASSES = 10
LEARNING_RATE = 1e-4

# Training hyperparameters
EPOCHS = 30
BATCH_SIZE = 32
VALIDATION_SIZE = 0.2
VALIDATION_TO_TEST_RATIO = 0.5
RANDOM_STATE = 42

# Data augmentation parameters
AUGMENTATIONS_PER_IMAGE = 4
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
SHEAR_RANGE = 0.1
ZOOM_RANGE = 0.1
HORIZONTAL_FLIP = True
FILL_MODE = "nearest"

# Training callbacks
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.2
MIN_LR = 0.0001

# Label processing
LABEL_OFFSET = 1

"""Configuration constants and hyperparameters for butterfly classification."""

import os
import sys
from pathlib import Path


def _source_project_root() -> Path:
    """Return the repository root when running from source."""
    return Path(__file__).resolve().parent.parent


def _frozen_runtime_root() -> Path:
    """Return a stable writable directory for PyInstaller executables."""
    app_dir_name = "Butterfly-Image-Classification"

    if os.name == "nt":
        base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base_dir = Path.home() / "Library" / "Application Support"
    else:
        base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    return base_dir / app_dir_name


PROJECT_ROOT = _source_project_root()
RUNTIME_ROOT = _frozen_runtime_root() if getattr(sys, "frozen", False) else PROJECT_ROOT
DATA_DIR = RUNTIME_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = RUNTIME_ROOT / "models"
RESULTS_DIR = RUNTIME_ROOT / "results"

ZENODO_URL = "https://zenodo.org/records/7559420/files/leedsbutterfly_dataset_v1.1.zip?download=1"

IMAGE_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)

NUM_CLASSES = 10
LEARNING_RATE = 1e-4

EPOCHS = 30
BATCH_SIZE = 32
VALIDATION_SIZE = 0.2
VALIDATION_TO_TEST_RATIO = 0.5
RANDOM_STATE = 42

AUGMENTATIONS_PER_IMAGE = 4
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
SHEAR_RANGE = 0.1
ZOOM_RANGE = 0.1
HORIZONTAL_FLIP = True
FILL_MODE = "nearest"

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.2
MIN_LR = 1e-6

LABEL_OFFSET = 1

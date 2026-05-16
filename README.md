# Butterfly Image Classification

[![License: MIT](https://img.shields.io/github/license/gurovamr/Butterfly-Image-Classification)](LICENSE)
[![CI](https://github.com/gurovamr/Butterfly-Image-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/gurovamr/Butterfly-Image-Classification/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gurovamr/Butterfly-Image-Classification/branch/main/graph/badge.svg)](https://codecov.io/gh/gurovamr/Butterfly-Image-Classification)
[![Release](https://img.shields.io/github/v/release/gurovamr/Butterfly-Image-Classification)](https://github.com/gurovamr/Butterfly-Image-Classification/releases)

A deep learning pipeline that trains CNN models to classify 10 species of butterflies from the [Leeds Butterfly Dataset](https://zenodo.org/records/7559420). The project covers the full ML workflow: data download, preprocessing, augmentation, training two model architectures, evaluation and visualization.

---

## Features

- Automatic dataset download from Zenodo
- Image preprocessing, augmentation, normalization, and train/validation/test splitting
- Two CNN architectures: **baseline** and **improved** with batch normalization and dropout
- Per-class accuracy, confusion matrix, prediction samples, and training history plots
- Pre-built binaries for Linux, Windows, and macOS in [Releases](../../releases)

---

## Quick Start

**Clone and install:**

```bash
git clone https://github.com/gurovamr/Butterfly-Image-Classification.git
cd Butterfly-Image-Classification

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -e .
butterfly-classifier
```

**Run directly:**

```bash
python main.py
```

The first run automatically downloads and extracts the dataset. When running from source, files are stored in `data/images/`, `models/`, and `results/` inside the repository. When running a packaged executable, files are stored in your user app data directory instead, for example `%LOCALAPPDATA%\Butterfly-Image-Classification\` on Windows.

---

## Installation For Development

```bash
python -m pip install --upgrade pip
pip install -e ".[dev,test]"
```

This installs the project in editable mode with runtime dependencies, testing tools and linting/formatting tools.

Before running the full training pipeline, verify that TensorFlow can load in the active
environment:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

If this fails on Windows with a TensorFlow DLL or native runtime error, recreate the
environment instead of installing into an existing base Anaconda environment. Also make
sure the Microsoft Visual C++ Redistributable is installed. The project supports Python
3.10 through 3.13, matching the TensorFlow version used by this project.

---

## Running Tests

```bash
pytest
```

Coverage is enforced at 80%. To see a detailed coverage report:

```bash
pytest --cov=scripts --cov-report=term-missing
```

---

## Project Structure

```text
Butterfly-Image-Classification/
|-- main.py                         # Script for the full workflow
|-- scripts/                        # Core scripts
|   |-- __init__.py                 # Package marker and version
|   |-- config.py                   # Hyperparameters and path constants
|   |-- data_download.py            # Dataset download, extraction and cleanup
|   |-- data_preprocessing.py       # Image loading, augmentation, normalization and splitting
|   |-- model.py                    # CNN model architectures
|   |-- train.py                    # Training loop and callbacks
|   |-- evaluator.py                # Evaluation, predictions, and per-class accuracy
|   `-- visualizer.py               # Plot generation for training and evaluation results
|-- tests/                          # pytest test suite
|   |-- conftest.py                 # Shared fixtures
|   |-- test_data_download.py
|   |-- test_data_preprocessing.py
|   |-- test_evaluator.py
|   |-- test_model_train.py
|   `-- test_visualizer.py
|-- project_documents/              # Project assignments
|   `-- Final_Presentation.pdf      
|-- .github/
|   |-- workflows/
|   |   |-- ci.yml                  # Linting, formatting, tests, coverage and Codecov
|   |   |-- matrix.yml              # Python version matrix and pylint
|   |   `-- build_binaries.yml      # PyInstaller release builds
|   `-- dependabot.yml              # Dependency update automation
|-- pyproject.toml                  # Package dependencies
|-- codecov.yml                     # Codecov status-check configuration
`-- LICENSE
```

---

## Configuration

All tunable parameters live in [`scripts/config.py`](scripts/config.py):

| Parameter | Default | Description |
|---|---:|---|
| `IMAGE_SIZE` | `(128, 128)` | Input image resolution |
| `NUM_CLASSES` | `10` | Number of butterfly species |
| `EPOCHS` | `30` | Training epochs |
| `BATCH_SIZE` | `32` | Mini-batch size |
| `AUGMENTATIONS_PER_IMAGE` | `4` | Augmented copies per original image |
| `VALIDATION_SIZE` | `0.2` | Fraction held out for validation and test data |
| `LEARNING_RATE` | `1e-4` | Adam learning rate |

---

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| CI | Push to `main`, every PR | Ruff, Black, tests, coverage upload |
| Matrix | Push to `main`, every PR, manual dispatch | Tests on Python 3.10, 3.11, and 3.12 plus pylint |
| Build Binaries | `v*` tag push, manual dispatch | Builds executables for Linux, Windows, and macOS |

---

## Requirements

- Python >= 3.10 and < 3.14
- TensorFlow 2.21+
- See [`pyproject.toml`](pyproject.toml) for the full dependency list

---

## Authors

- Mariia Gurova
- Pia Lagler
- Kai Aebli

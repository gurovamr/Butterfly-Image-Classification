# Butterfly Image Classification

[![License: MIT](https://img.shields.io/github/license/gurovamr/Butterfly-Image-Classification)](LICENSE)
[![CI](https://github.com/gurovamr/Butterfly-Image-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/gurovamr/Butterfly-Image-Classification/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gurovamr/Butterfly-Image-Classification/graph/badge.svg)](https://codecov.io/gh/gurovamr/Butterfly-Image-Classification)
[![Release](https://img.shields.io/github/v/release/gurovamr/Butterfly-Image-Classification)](https://github.com/gurovamr/Butterfly-Image-Classification/releases)

A deep learning pipeline that trains CNN models to classify 10 species of butterflies from the [Leeds Butterfly Dataset](https://zenodo.org/records/7559420). The project covers the full ML workflow: data download, augmentation, training two model architectures, evaluation, and visualization.

---

## Features

- Automatic dataset download from Zenodo (no manual steps)
- Image augmentation pipeline (rotation, shift, zoom, flip)
- Two CNN architectures: **baseline** and **improved** (with batch normalization and dropout)
- Per-class accuracy, confusion matrix, and training history plots
- Pre-built binaries for Linux, Windows, and macOS (see [Releases](../../releases))

---

## Quick Start

**Clone and install:**
```bash
# 1. clone repo
git clone https://github.com/gurovamr/Butterfly-Image-Classification.git
cd Butterfly-Image-Classification

# 2. create environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# 3. install project
pip install -e .

# 4. run
butterfly-classifier
```

**Run the full pipeline:**
```bash
butterfly-classifier
```

Or directly:
```bash
python main.py
```

The first run automatically downloads and extracts the dataset (~50 MB) into `data/images/`. Trained models are saved to `models/` and result plots to `results/`.

---

## Installation (Development)

```bash
pip install ".[dev]"
```

This installs all runtime dependencies plus testing and linting tools (pytest, ruff, black, pylint).

---

## Running Tests

```bash
pytest
```

Coverage is enforced at 80%. To see a report:
```bash
pytest --cov=scripts --cov-report=term-missing
```

---

## Project Structure

```
Butterfly-Image-Classification/
│
├── main.py                   # Entry point — runs the full pipeline
│
├── scripts/                  # Core library
│   ├── __init__.py           # Package marker; exposes __version__
│   ├── config.py             # All hyperparameters and path constants
│   ├── data_download.py      # Dataset download, extraction, cleanup
│   ├── dataset.py            # Image loading, augmentation, normalisation, splitting
│   ├── model.py              # CNN architectures (baseline & improved)
│   ├── train.py              # Training loop, callbacks (EarlyStopping, ReduceLR)
│   ├── evaluator.py          # Evaluation, prediction, per-class accuracy
│   └── visualizer.py         # All matplotlib plots saved to results/
│
├── tests/                    # pytest test suite
│   ├── conftest.py           # Shared fixtures (make_image factory)
│   ├── test_data_download.py
│   ├── test_dataset.py
│   ├── test_evaluator.py
│   ├── test_model_train.py
│   └── test_visualizer.py
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml            # Lint (ruff, black) + test + coverage → Codecov
│   │   ├── matrix.yml        # Tests on Python 3.10 / 3.11 / 3.12 + pylint
│   │   └── build_binaries.yml# PyInstaller builds for Linux/Windows/macOS on tag push
│   └── dependabot.yml        # Weekly dependency updates
│
├── pyproject.toml            # Package config, dependencies, tool config (black, ruff, pylint, pytest, coverage)
└── codecov.yml               # Codecov PR status check config
```

---

## Configuration

All tunable parameters live in [`scripts/config.py`](scripts/config.py):

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | `(128, 128)` | Input image resolution |
| `NUM_CLASSES` | `10` | Number of butterfly species |
| `EPOCHS` | `30` | Training epochs |
| `BATCH_SIZE` | `32` | Mini-batch size |
| `AUGMENTATIONS_PER_IMAGE` | `4` | Augmented copies per original image |
| `VALIDATION_SIZE` | `0.2` | Fraction held out for val + test |
| `LEARNING_RATE` | `1e-4` | Adam learning rate |

---

## CI / CD

| Workflow | Trigger | What it does |
|---|---|---|
| CI | Push to `main`, every PR | Lint → tests → coverage upload |
| Matrix | Push to `main`, every PR | Tests on Python 3.10 / 3.11 / 3.12 |
| Build Binaries | `v*` tag push | Builds executables for all platforms and creates a GitHub Release |

---

## Requirements

- Python ≥ 3.10
- TensorFlow 2.21+
- See `pyproject.toml` for the full pinned dependency list

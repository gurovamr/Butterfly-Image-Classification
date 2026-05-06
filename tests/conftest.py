import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def make_image():
    """Factory fixture that returns a helper to create fake image arrays."""

    def _make(h: int = 128, w: int = 128, c: int = 3, fill: float = 128.0) -> np.ndarray:
        return np.full((h, w, c), fill, dtype=np.float32)

    return _make

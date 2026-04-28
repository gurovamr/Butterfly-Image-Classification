import pytest
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from scripts.model import ButterflyClassifier
from scripts.train import ModelTrainer


class DummyFitModel:
    def __init__(self):
        self.received_kwargs = None

    def fit(self, train_images, train_labels, **kwargs):
        self.received_kwargs = kwargs
        return "history"


class TestModelTrainer:
    # Equivalence class 1: callbacks reflect the configured patience and factor values
    def test_create_callbacks_uses_configured_values(self):
        trainer = ModelTrainer(
            early_stopping_patience=7,
            reduce_lr_patience=4,
            reduce_lr_factor=0.5,
            min_lr=1e-5,
        )

        callbacks = trainer.create_callbacks()

        assert len(callbacks) == 2
        assert isinstance(callbacks[0], EarlyStopping)
        assert callbacks[0].patience == 7
        assert callbacks[0].restore_best_weights is True
        assert isinstance(callbacks[1], ReduceLROnPlateau)
        assert callbacks[1].patience == 4
        # Use pytest.approx for floats — exact equality is a pitfall with floating point
        assert callbacks[1].factor == pytest.approx(0.5)
        assert callbacks[1].min_lr == pytest.approx(1e-5)


class TestButterflyClassifier:
    # Equivalence class 1: valid model types build without error
    def test_builds_baseline_model(self):
        classifier = ButterflyClassifier(model_type="baseline")
        assert classifier.model is not None

    def test_builds_improved_model(self):
        classifier = ButterflyClassifier(model_type="improved")
        assert classifier.model is not None

    # Equivalence class 2: edge case — unknown model type raises ValueError
    def test_raises_for_unknown_model_type(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            ButterflyClassifier(model_type="unsupported")
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

    # Equivalence class 3: save() delegates to model.save with a posix path
    def test_save_calls_model_save(self, tmp_path, mocker):
        classifier = ButterflyClassifier(model_type="baseline")
        mock_save = mocker.patch.object(classifier.model, "save")
        classifier.save(tmp_path / "model.keras")
        mock_save.assert_called_once()

    # Equivalence class 4: load() returns whatever load_model returns
    def test_load_returns_model(self, mocker):
        mock_model = object()
        mocker.patch("scripts.model.load_model", return_value=mock_model)
        result = ButterflyClassifier.load("/fake/path/model.keras")
        assert result is mock_model


class TestModelTrainerTrain:
    # Equivalence class 1: train() with explicit callbacks passes them through to model.fit
    def test_train_with_explicit_callbacks(self):
        import numpy as np

        trainer = ModelTrainer()
        dummy_model = DummyFitModel()
        train_images = np.zeros((4, 8, 8, 3), dtype=np.float32)
        train_labels = np.array([0, 1, 0, 1], dtype=np.int32)
        val_images = np.zeros((2, 8, 8, 3), dtype=np.float32)
        val_labels = np.array([0, 1], dtype=np.int32)
        callbacks = []

        result = trainer.train(
            dummy_model,
            train_images,
            train_labels,
            val_images,
            val_labels,
            epochs=1,
            callbacks=callbacks,
        )

        assert result == "history"
        assert dummy_model.received_kwargs["epochs"] == 1
        assert dummy_model.received_kwargs["callbacks"] is callbacks

    # Equivalence class 2: train() with callbacks=None — default callbacks are created automatically
    def test_train_creates_default_callbacks_when_none(self):
        import numpy as np

        trainer = ModelTrainer()
        dummy_model = DummyFitModel()
        train_images = np.zeros((4, 8, 8, 3), dtype=np.float32)
        train_labels = np.array([0, 1, 0, 1], dtype=np.int32)
        val_images = np.zeros((2, 8, 8, 3), dtype=np.float32)
        val_labels = np.array([0, 1], dtype=np.int32)

        result = trainer.train(
            dummy_model, train_images, train_labels, val_images, val_labels, epochs=2
        )

        assert result == "history"
        assert len(dummy_model.received_kwargs["callbacks"]) == 2

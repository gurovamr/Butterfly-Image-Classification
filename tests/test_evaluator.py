import numpy as np
import pytest

from scripts.evaluator import ModelEvaluator


class DummyEvaluateModel:
    def evaluate(self, images: np.ndarray, labels: np.ndarray):
        return 0.25, 0.9


class DummyPredictionModel:
    def predict(self, images: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0.1, 0.7, 0.2],
                [0.8, 0.1, 0.1],
                [0.2, 0.3, 0.5],
            ],
            dtype=np.float32,
        )


class TestPredict:
    # Equivalence class 1: argmax of softmax probabilities gives correct class indices
    def test_returns_argmax_class_indices(self):
        images = np.zeros((3, 8, 8, 3), dtype=np.float32)

        predicted = ModelEvaluator.predict(DummyPredictionModel(), images)

        assert np.array_equal(predicted, np.array([1, 0, 2]))


class TestCalculateClassAccuracy:
    # Equivalence class 1: mixed correct/incorrect predictions
    def test_mixed_predictions_report_correct_accuracy_per_class(self):
        evaluator = ModelEvaluator(num_classes=4)
        true_labels = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        predicted_labels = np.array([0, 1, 1, 2, 0], dtype=np.int32)

        result = evaluator.calculate_class_accuracy(true_labels, predicted_labels)

        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(0.5)

    # Equivalence class 2: edge case — class with no samples returns 0.0
    def test_class_with_no_samples_returns_zero_accuracy(self):
        evaluator = ModelEvaluator(num_classes=4)
        true_labels = np.array([0, 1, 2], dtype=np.int32)
        predicted_labels = np.array([0, 1, 2], dtype=np.int32)

        result = evaluator.calculate_class_accuracy(true_labels, predicted_labels)

        assert result[3] == pytest.approx(0.0)

    # Equivalence class 3: all predictions correct — every class returns 1.0
    def test_perfect_predictions_give_1_0(self):
        evaluator = ModelEvaluator(num_classes=3)
        labels = np.array([0, 1, 2], dtype=np.int32)
        result = evaluator.calculate_class_accuracy(labels, labels)
        assert all(result[c] == pytest.approx(1.0) for c in range(3))


class TestEvaluate:
    # Equivalence class 1: evaluate() returns loss and accuracy unpacked as floats
    def test_returns_loss_and_accuracy_as_floats(self):
        images = np.zeros((3, 8, 8, 3), dtype=np.float32)
        labels = np.array([0, 1, 2], dtype=np.int32)
        evaluator = ModelEvaluator(num_classes=3)
        loss, acc = evaluator.evaluate(DummyEvaluateModel(), images, labels)
        assert loss == pytest.approx(0.25)
        assert acc == pytest.approx(0.9)

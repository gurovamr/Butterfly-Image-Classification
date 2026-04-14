import numpy as np

from scripts.evaluater import ModelEvaluator


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
    def test_returns_argmax_class_indices(self):
        images = np.zeros((3, 8, 8, 3), dtype=np.float32)

        predicted = ModelEvaluator.predict(DummyPredictionModel(), images)

        assert np.array_equal(predicted, np.array([1, 0, 2]))


class TestCalculateClassAccuracy:
    def test_reports_accuracy_per_class_and_zero_for_missing_classes(self):
        evaluator = ModelEvaluator(num_classes=4)
        true_labels = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        predicted_labels = np.array([0, 1, 1, 2, 0], dtype=np.int32)

        result = evaluator.calculate_class_accuracy(true_labels, predicted_labels)

        assert result[0] == 0.5
        assert result[1] == 1.0
        assert result[2] == 0.5
        assert result[3] == 0.0
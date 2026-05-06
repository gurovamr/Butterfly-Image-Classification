"""Model evaluation, prediction, and metrics calculation utilities."""

from __future__ import annotations

from collections import Counter

import numpy as np
from tensorflow.keras.models import Sequential


class ModelEvaluator:
    """Handles model evaluation, prediction, and metrics calculation.

    Provides methods for model evaluation on test sets, prediction,
    and detailed per-class accuracy analysis.
    """

    def __init__(self, num_classes: int = 10):
        """Initialize evaluator.

        Parameters
        ----------
        num_classes : int, default=10
                Number of classification classes.
        """
        self.num_classes = num_classes

    def evaluate(
        self,
        model: Sequential,
        test_images: np.ndarray,
        test_labels: np.ndarray,
    ) -> tuple[float, float]:
        """Evaluate model on test data.

        Parameters
        ----------
        model : Sequential
                Trained Keras model to evaluate.
        test_images : np.ndarray
                Test images (normalized).
        test_labels : np.ndarray
                True test labels.

        Returns
        -------
        tuple[float, float]
                Tuple of (test_loss, test_accuracy).
        """
        test_loss, test_accuracy = model.evaluate(test_images, test_labels)
        return float(test_loss), float(test_accuracy)

    @staticmethod
    def predict(model: Sequential, images: np.ndarray) -> np.ndarray:
        """Predict class labels for input images.

        Parameters
        ----------
        model : Sequential
                Trained Keras model.
        images : np.ndarray
                Input images to predict on (normalized).

        Returns
        -------
        np.ndarray
                Predicted class indices (argmax of softmax probabilities).
        """
        predictions = model.predict(images)
        return np.argmax(predictions, axis=1)

    def calculate_class_accuracy(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> dict[int, float]:
        """Calculate per-class accuracy metrics.

        Parameters
        ----------
        true_labels : np.ndarray
                Ground truth labels.
        predicted_labels : np.ndarray
                Predicted labels from model.

        Returns
        -------
        dict[int, float]
                Dictionary mapping class index to accuracy (0.0 to 1.0).
                Classes with no samples return 0.0 accuracy.
        """
        correct_predictions = predicted_labels == true_labels
        correct_counts: Counter[int] = Counter()
        total_counts: Counter[int] = Counter()

        for true_label, is_correct in zip(true_labels, correct_predictions):
            total_counts[int(true_label)] += 1
            if bool(is_correct):
                correct_counts[int(true_label)] += 1

        accuracy_per_class: dict[int, float] = {}
        for class_label in range(self.num_classes):
            total = total_counts[class_label]
            correct = correct_counts[class_label]
            accuracy_per_class[class_label] = (correct / total) if total > 0 else 0.0

        return accuracy_per_class

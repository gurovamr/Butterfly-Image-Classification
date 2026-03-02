"""Model construction, training, and evaluation utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
	BatchNormalization,
	Conv2D,
	Dense,
	Dropout,
	Flatten,
	MaxPooling2D,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


def create_training_callbacks() -> list[Callback]:
	"""Create default callbacks used in notebook experiments.

	Returns
	-------
	list[Callback]
		Early stopping and learning-rate reduction callbacks.
	"""
	early_stopping = EarlyStopping(
		monitor="val_loss",
		patience=5,
		restore_best_weights=True,
	)
	reduce_lr = ReduceLROnPlateau(
		monitor="val_loss",
		factor=0.2,
		patience=3,
		min_lr=0.0001,
	)
	return [early_stopping, reduce_lr]


def build_baseline_model(
	input_shape: tuple[int, int, int] = (128, 128, 3),
	num_classes: int = 10,
	learning_rate: float = 1e-4,
) -> Sequential:
	"""Build and compile the baseline CNN from the notebook."""
	model = Sequential(
		[
			Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
			MaxPooling2D(2, 2),
			Conv2D(64, (3, 3), activation="relu"),
			BatchNormalization(),
			MaxPooling2D(2, 2),
			Conv2D(128, (3, 3), activation="relu"),
			MaxPooling2D(2, 2),
			Flatten(),
			Dense(256, activation="relu"),
			Dense(128, activation="relu"),
			Dense(num_classes, activation="softmax"),
		]
	)
	model.compile(
		optimizer=Adam(learning_rate=learning_rate),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def build_improved_model(
	input_shape: tuple[int, int, int] = (128, 128, 3),
	num_classes: int = 10,
	learning_rate: float = 1e-4,
) -> Sequential:
	"""Build and compile the improved CNN with dropout and extra normalization."""
	model = Sequential(
		[
			Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
			BatchNormalization(),
			MaxPooling2D(2, 2),
			Conv2D(64, (3, 3), activation="relu"),
			BatchNormalization(),
			MaxPooling2D(2, 2),
			Conv2D(128, (3, 3), activation="relu"),
			BatchNormalization(),
			MaxPooling2D(2, 2),
			Flatten(),
			Dense(256, activation="relu"),
			Dropout(0.2),
			Dense(128, activation="relu"),
			Dropout(0.3),
			Dense(num_classes, activation="softmax"),
		]
	)
	model.compile(
		optimizer=Adam(learning_rate=learning_rate),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def train_model(
	model: Sequential,
	train_images: np.ndarray,
	train_labels: np.ndarray,
	val_images: np.ndarray,
	val_labels: np.ndarray,
	epochs: int = 30,
	callbacks: list[Callback] | None = None,
) -> tf.keras.callbacks.History:
	"""Train a model and return the Keras history object."""
	return model.fit(
		train_images,
		train_labels,
		epochs=epochs,
		validation_data=(val_images, val_labels),
		callbacks=callbacks or [],
	)


def evaluate_model(
	model: Sequential,
	test_images: np.ndarray,
	test_labels: np.ndarray,
) -> tuple[float, float]:
	"""Evaluate model on test data and return `(loss, accuracy)`."""
	test_loss, test_accuracy = model.evaluate(test_images, test_labels)
	return float(test_loss), float(test_accuracy)


def save_model_artifact(model: Sequential, output_path: str | Path) -> None:
	"""Save a trained Keras model to disk."""
	model.save(Path(output_path).as_posix())


def load_model_artifact(model_path: str | Path) -> Sequential:
	"""Load a Keras model from disk."""
	return load_model(Path(model_path).as_posix())


def predict_class_labels(model: Sequential, images: np.ndarray) -> np.ndarray:
	"""Predict class indices for image tensors."""
	predictions = model.predict(images)
	return np.argmax(predictions, axis=1)


def calculate_class_accuracy(
	true_labels: np.ndarray,
	predicted_labels: np.ndarray,
	num_classes: int = 10,
) -> dict[int, float]:
	"""Calculate per-class accuracy dictionary."""
	correct_predictions = predicted_labels == true_labels
	correct_counts: Counter[int] = Counter()
	total_counts: Counter[int] = Counter()

	for true_label, is_correct in zip(true_labels, correct_predictions):
		total_counts[int(true_label)] += 1
		if bool(is_correct):
			correct_counts[int(true_label)] += 1

	accuracy_per_class: dict[int, float] = {}
	for class_label in range(num_classes):
		total = total_counts[class_label]
		correct = correct_counts[class_label]
		accuracy_per_class[class_label] = (correct / total) if total > 0 else 0.0

	return accuracy_per_class
"""Visualization utilities for training and evaluation outputs."""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Use non-interactive backend to prevent popups
matplotlib.use('Agg')


class TrainingVisualizer:
	"""Visualization utilities for model training and evaluation.
	
	Saves all plots to a specified directory instead of displaying them.
	Provides methods for plotting training history, predictions,
	confusion matrices, and data exploration visualizations.
	"""

	def __init__(self, num_classes: int = 10, save_dir: str | Path = "results"):
		"""Initialize visualizer with output directory for saving plots.
		
		Parameters
		----------
		num_classes : int, default=10
			Number of classification classes.
		save_dir : str | Path, default="results"
			Directory where plots will be saved.
		"""
		self.num_classes = num_classes
		self.save_dir = Path(save_dir)
		self.save_dir.mkdir(parents=True, exist_ok=True)
		self.plot_counter = 0  # Track plot numbers for unique filenames

	def plot_random_images(
		self,
		original_images: list[np.ndarray],
		augmented_images: list[np.ndarray],
		original_labels: list[int],
		augmented_labels: list[int],
		num_images: int = 5,
	) -> None:
		"""Plot random original and augmented images side by side.
		
		Parameters
		----------
		original_images : list[np.ndarray]
			List of original images.
		augmented_images : list[np.ndarray]
			List of augmented images.
		original_labels : list[int]
			Labels for original images.
		augmented_labels : list[int]
			Labels for augmented images.
		num_images : int, default=5
			Number of image pairs to display.
		"""
		plt.figure(figsize=(20, 10))
		original_indices = random.sample(
			range(len(original_images)), k=min(num_images, len(original_images))
		)
		for row_index, image_index in enumerate(original_indices):
			plt.subplot(2, num_images, row_index + 1)
			plt.imshow(original_images[image_index].astype("uint8"))
			plt.title(f"Original Label: {original_labels[image_index]}")
			plt.axis("off")

		augmented_indices = random.sample(
			range(len(augmented_images)),
			k=min(num_images, len(augmented_images)),
		)
		for row_index, image_index in enumerate(augmented_indices):
			plt.subplot(2, num_images, num_images + row_index + 1)
			plt.imshow(augmented_images[image_index].astype("uint8"))
			plt.title(f"Augmented Label: {augmented_labels[image_index]}")
			plt.axis("off")

		plt.tight_layout()
		output_path = self.save_dir / "01_random_images_comparison.png"
		plt.savefig(output_path, dpi=150, bbox_inches='tight')
		plt.close()
		print(f"Saved: {output_path}")

	def plot_class_examples(
		self,
		images: list[np.ndarray],
		labels: list[int],
		class_index: int = 0,
		num_images: int = 5,
	) -> None:
		"""Plot example images from a specific class.
		
		Parameters
		----------
		images : list[np.ndarray]
			List of images.
		labels : list[int]
			Corresponding labels.
		class_index : int, default=0
			Index of class to display (in sorted unique labels).
		num_images : int, default=5
			Number of examples to show.
			
		Raises
		------
		ValueError
			If class_index is out of range.
		"""
		unique_labels = sorted(set(labels))
		if class_index < 0 or class_index >= len(unique_labels):
			raise ValueError("class_index is out of range for available labels")

		target_label = unique_labels[class_index]
		target_images = [
			image for image, label in zip(images, labels) if label == target_label
		]
		num_to_plot = min(num_images, len(target_images))

		plt.figure(figsize=(20, 5))
		for index in range(num_to_plot):
			plt.subplot(1, num_to_plot, index + 1)
			plt.imshow(target_images[index].astype("uint8"))
			plt.title(f"Class {target_label}")
			plt.axis("off")
		plt.suptitle(f"{num_to_plot} Images from Class {target_label}")
		plt.tight_layout()
		output_path = self.save_dir / f"02_class_{target_label}_examples.png"
		plt.savefig(output_path, dpi=150, bbox_inches='tight')
		plt.close()
		print(f"Saved: {output_path}")

	def plot_training_history(
		self,
		history: dict[str, list[float]], title_suffix: str = ""
	) -> None:
		"""Plot training and validation loss/accuracy curves.
		
		Parameters
		----------
		history : dict[str, list[float]]
			Keras history dictionary with keys: loss, val_loss, accuracy, val_accuracy.
		title_suffix : str, default=""
			Optional suffix to add to plot titles.
		"""
		# Sanitize suffix for filename
		filename_suffix = title_suffix.replace(" ", "_").replace("(", "").replace(")", "").lower()
		
		plt.figure(figsize=(10, 6))
		plt.plot(history.get("loss", []), color="teal", label="Training Loss")
		plt.plot(history.get("val_loss", []), color="orange", label="Validation Loss")
		plt.title(f"Loss Over Epochs{title_suffix}", fontsize=20)
		plt.xlabel("Epochs", fontsize=14)
		plt.ylabel("Loss", fontsize=14)
		plt.legend()
		plt.grid(True)
		output_path = self.save_dir / f"03_loss{filename_suffix}.png"
		plt.savefig(output_path, dpi=150, bbox_inches='tight')
		plt.close()
		print(f"Saved: {output_path}")

		plt.figure(figsize=(10, 6))
		plt.plot(history.get("accuracy", []), color="teal", label="Training Accuracy")
		plt.plot(
			history.get("val_accuracy", []), color="orange", label="Validation Accuracy"
		)
		plt.title(f"Accuracy Over Epochs{title_suffix}", fontsize=20)
		plt.xlabel("Epochs", fontsize=14)
		plt.ylabel("Accuracy", fontsize=14)
		plt.legend()
		plt.grid(True)
		output_path = self.save_dir / f"04_accuracy{filename_suffix}.png"
		plt.savefig(output_path, dpi=150, bbox_inches='tight')
		plt.close()
		print(f"Saved: {output_path}")

	def plot_confusion_matrix(
		self,
		true_labels: np.ndarray,
		predicted_labels: np.ndarray,
	) -> None:
		"""Render confusion matrix heatmap.
		
		Parameters
		----------
		true_labels : np.ndarray
			Ground truth labels.
		predicted_labels : np.ndarray
			Predicted labels.
		"""
		conf_matrix = confusion_matrix(true_labels, predicted_labels)
		plt.figure(figsize=(10, 8))
		sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
		plt.xlabel("Predicted Labels")
		plt.ylabel("True Labels")
		plt.title("Confusion Matrix")
		output_path = self.save_dir / "05_confusion_matrix.png"
		plt.savefig(output_path, dpi=150, bbox_inches='tight')
		plt.close()
		print(f"Saved: {output_path}")

	def plot_prediction_samples(
		self,
		images: np.ndarray,
		true_labels: np.ndarray,
		predicted_labels: np.ndarray,
		num_images: int = 10,
	) -> None:
		"""Plot random prediction samples with true and predicted labels.
		
		Parameters
		----------
		images : np.ndarray
			Image array.
		true_labels : np.ndarray
			Ground truth labels.
		predicted_labels : np.ndarray
			Model predictions.
		num_images : int, default=10
			Number of samples to display.
		"""
		num_samples = min(num_images, len(images))
		selected_indices = random.sample(range(len(images)), k=num_samples)
		selected_images = images[selected_indices]
		selected_true_labels = true_labels[selected_indices]
		selected_pred_labels = predicted_labels[selected_indices]

		plt.figure(figsize=(20, 10))
		for index in range(num_samples):
			plt.subplot(2, math.ceil(num_samples / 2), index + 1)
			plt.imshow(selected_images[index])
			plt.title(
				f"True: {selected_true_labels[index]}, Pred: {selected_pred_labels[index]}"
			)
			plt.axis("off")
		plt.tight_layout()
		output_path = self.save_dir / "06_prediction_samples.png"
		plt.savefig(output_path, dpi=150, bbox_inches='tight')
		plt.close()
		print(f"Saved: {output_path}")

	def plot_class_histogram(
		self,
		true_labels: np.ndarray,
		predicted_labels: np.ndarray,
	) -> None:
		"""Plot histogram of correct vs total predictions per class.
		
		Parameters
		----------
		true_labels : np.ndarray
			Ground truth labels.
		predicted_labels : np.ndarray
			Model predictions.
		"""
		correct_mask = predicted_labels == true_labels
		correct_counts: Counter[int] = Counter()
		total_counts: Counter[int] = Counter()

		for true_label, is_correct in zip(true_labels, correct_mask):
			total_counts[int(true_label)] += 1
			if bool(is_correct):
				correct_counts[int(true_label)] += 1

		classes = list(range(self.num_classes))
		correct_values = [correct_counts[class_label] for class_label in classes]
		total_values = [total_counts[class_label] for class_label in classes]

		plt.figure(figsize=(10, 6))
		plt.bar(
			classes, correct_values, color="blue", alpha=0.6, label="Correct Predictions"
		)
		plt.bar(
			classes, total_values, color="red", alpha=0.3, label="Total Predictions"
		)
		plt.xlabel("Class Labels")
		plt.ylabel("Number of Predictions")
		plt.title("Correct Predictions per Class")
		plt.legend()
		plt.xticks(classes)
		output_path = self.save_dir / "07_class_prediction_histogram.png"
		plt.savefig(output_path, dpi=150, bbox_inches='tight')
		plt.close()
		print(f"Saved: {output_path}")
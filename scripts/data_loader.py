"""Data loading, augmentation, and split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


@dataclass(frozen=True)
class DatasetSplits:
	"""Container for train, validation, and test splits."""

	train_images: np.ndarray
	val_images: np.ndarray
	test_images: np.ndarray
	train_labels: np.ndarray
	val_labels: np.ndarray
	test_labels: np.ndarray


def load_labeled_images_from_directory(
	images_directory: str | Path,
	image_size: tuple[int, int] = (128, 128),
) -> tuple[list[np.ndarray], list[int]]:
	"""Load images and labels inferred from filename prefixes.

	The label extraction follows the original notebook convention where
	characters at index `1:3` in the filename stem represent the class ID.
	Example: `E03_001.jpg` -> label `3`.

	Parameters
	----------
	images_directory:
		Path to the folder containing butterfly image files.
	image_size:
		Target size used when loading each image as `(height, width)`.

	Returns
	-------
	tuple[list[np.ndarray], list[int]]
		A tuple with image arrays and aligned integer labels.
	"""
	directory = Path(images_directory)
	filenames = sorted(path.name for path in _iter_files(directory))

	categories: dict[str, int] = {}
	for filename in filenames:
		image_stem = Path(filename).stem
		if image_stem[1:3].isdigit():
			categories[filename] = int(image_stem[1:3])

	images: list[np.ndarray] = []
	labels: list[int] = []
	for filename, label in categories.items():
		image_path = directory / filename
		image = load_img(image_path, target_size=image_size)
		images.append(img_to_array(image))
		labels.append(label)

	return images, labels


def create_augmented_dataset(
	images: list[np.ndarray],
	labels: list[int],
	augmentations_per_image: int = 4,
) -> tuple[list[np.ndarray], list[int]]:
	"""Generate augmented images with labels using Keras ImageDataGenerator.

	Parameters
	----------
	images:
		Original image arrays.
	labels:
		Original labels aligned with `images`.
	augmentations_per_image:
		How many augmented samples to generate per original image.

	Returns
	-------
	tuple[list[np.ndarray], list[int]]
		Augmented image arrays and aligned labels.
	"""
	datagen = ImageDataGenerator(
		rotation_range=15,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.1,
		zoom_range=0.1,
		horizontal_flip=True,
		fill_mode="nearest",
	)

	augmented_images: list[np.ndarray] = []
	augmented_labels: list[int] = []

	for image_array, label in zip(images, labels):
		single_image_batch = np.expand_dims(image_array, axis=0)
		generator = datagen.flow(single_image_batch, batch_size=1)
		for _ in range(augmentations_per_image):
			augmented_image = next(generator)[0].astype(np.float32)
			augmented_images.append(augmented_image)
			augmented_labels.append(label)

	return augmented_images, augmented_labels


def combine_and_normalize_images(
	original_images: list[np.ndarray],
	original_labels: list[int],
	augmented_images: list[np.ndarray],
	augmented_labels: list[int],
) -> tuple[np.ndarray, np.ndarray]:
	"""Combine original + augmented samples and normalize pixel values.

	Parameters
	----------
	original_images:
		Original image arrays.
	original_labels:
		Labels for original images.
	augmented_images:
		Augmented image arrays.
	augmented_labels:
		Labels for augmented images.

	Returns
	-------
	tuple[np.ndarray, np.ndarray]
		Combined normalized images (float32 in [0, 1]) and labels.
	"""
	all_images = original_images + augmented_images
	all_labels = original_labels + augmented_labels

	all_images_array = np.asarray(all_images, dtype=np.float32)
	normalized_images = all_images_array / 255.0
	labels_array = np.asarray(all_labels, dtype=np.int32)

	return normalized_images, labels_array


def split_dataset(
	images: np.ndarray,
	labels: np.ndarray,
	validation_size: float = 0.2,
	validation_to_test_ratio: float = 0.5,
	random_state: int = 42,
	label_offset: int = 1,
) -> DatasetSplits:
	"""Split data into train, validation, and test sets.

	This reproduces notebook behavior where labels are decremented by one after
	splitting because source labels start at 1 while class indices should start at 0.

	Parameters
	----------
	images:
		Normalized image tensor.
	labels:
		Label array.
	validation_size:
		Fraction reserved from full data for val+test.
	validation_to_test_ratio:
		Fraction of val+test split used as test set.
	random_state:
		Random state for deterministic splitting.
	label_offset:
		Value subtracted from labels (default 1).

	Returns
	-------
	DatasetSplits
		Structured train/validation/test arrays.
	"""
	(
		train_images,
		val_images,
		train_labels,
		val_labels,
	) = train_test_split(
		images,
		labels,
		test_size=validation_size,
		random_state=random_state,
	)

	(
		val_images,
		test_images,
		val_labels,
		test_labels,
	) = train_test_split(
		val_images,
		val_labels,
		test_size=validation_to_test_ratio,
		random_state=random_state,
	)

	return DatasetSplits(
		train_images=train_images,
		val_images=val_images,
		test_images=test_images,
		train_labels=np.asarray(train_labels, dtype=np.int32) - label_offset,
		val_labels=np.asarray(val_labels, dtype=np.int32) - label_offset,
		test_labels=np.asarray(test_labels, dtype=np.int32) - label_offset,
	)


def _iter_files(directory: Path) -> Iterable[Path]:
	"""Yield all direct child files from a directory."""
	return (path for path in directory.iterdir() if path.is_file())
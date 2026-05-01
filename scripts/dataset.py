"""Data loading, augmentation, and split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import (
	ImageDataGenerator,
	img_to_array,
	load_img,
)


@dataclass(frozen=True)
class DatasetSplits:
	"""Container for train, validation, and test splits."""

	train_images: np.ndarray
	val_images: np.ndarray
	test_images: np.ndarray
	train_labels: np.ndarray
	val_labels: np.ndarray
	test_labels: np.ndarray


class ImageClassificationDataLoader:
	"""Load images and labels from filenames."""

	def __init__(self, directory: str | Path, image_size: Tuple[int, int] = (128, 128)):
		"""Init loader."""
		self.directory = Path(directory)
		self.image_size = image_size

	def _list_image_file_paths(self) -> List[Path]:
		"""Return all file paths."""
		return [file_path for file_path in self.directory.iterdir() if file_path.is_file()]

	def _extract_labels(self, paths: List[Path]) -> Dict[Path, int]:
		"""Map file paths to labels."""
		path_to_label: Dict[Path, int] = {}

		for file_path in paths:
			stem = file_path.stem
			if stem[1:3].isdigit():
				path_to_label[file_path] = int(stem[1:3])

		return path_to_label

	def _load_images(
		self,
		path_to_label: Dict[Path, int],
	) -> Tuple[List[np.ndarray], List[int]]:
		"""Load images and labels."""
		images: List[np.ndarray] = []
		labels: List[int] = []

		for file_path, label in path_to_label.items():
			image = load_img(file_path, target_size=self.image_size)
			images.append(img_to_array(image))
			labels.append(label)

		return images, labels
	
	def load_images(self) -> Tuple[List[np.ndarray], List[int]]:
		"""Full loading pipeline."""
		image_paths = self._list_image_file_paths()
		path_to_label = self._extract_labels(image_paths)
		return self._load_images(path_to_label)


class ImageAugmentor:
	"""Apply image augmentations."""

	def __init__(self, augmentations_per_image: int = 4):
		"""Init augmentor."""
		self.augmentations_per_image = augmentations_per_image
		self.datagen = ImageDataGenerator(
			rotation_range 		= 15,
			width_shift_range 	= 0.1,
			height_shift_range	= 0.1,
			shear_range			= 0.1,
			zoom_range			= 0.1,
			horizontal_flip		= True,
			fill_mode			= "nearest",
		)

	def _augment_single(
		self,
		image: np.ndarray,
		label: int,
	) -> Tuple[List[np.ndarray], List[int]]:
		"""Augment one image."""
		images, labels = [], []
		batch = np.expand_dims(image, axis=0)
		generator = self.datagen.flow(batch, batch_size=1)

		for number in range(self.augmentations_per_image):
			aug_img = next(generator)[0].astype(np.float32)
			images.append(aug_img)
			labels.append(label)

		return images, labels

	def augment_dataset(
		self,
		images: List[np.ndarray],
		labels: List[int],
	) -> Tuple[List[np.ndarray], List[int]]:
		"""Augment full dataset."""
		augmented_images, augmented_labels = [], []

		for image, label in zip(images, labels):
			imgs, lbls = self._augment_single(image, label)
			augmented_images.extend(imgs)
			augmented_labels.extend(lbls)

		return augmented_images, augmented_labels

	@staticmethod
	def combine_images(
		original_images: List[np.ndarray],
		original_labels: List[int],
		augmented_images: List[np.ndarray],
		augmented_labels: List[int],
	) -> Tuple[List[np.ndarray], List[int]]:
		"""Combine original and augmented image/label lists."""
		return original_images + augmented_images, original_labels + augmented_labels

	@staticmethod
	def normalize_images(images: list[np.ndarray]) -> np.ndarray:
		"""Normalize pixel values to [0, 1] range.
		
		Parameters
		----------
		images : list[np.ndarray]
			List of image arrays with pixel values in [0, 255] range.
		
		Returns
		-------
		np.ndarray
			Normalized image array with values in [0, 1] range.

		Examples
		--------
		>>> import numpy as np
		>>> imgs = [np.full((1, 1, 3), 255.0, dtype=np.float32)]
		>>> result = ImageAugmentor.normalize_images(imgs)
		>>> float(result.max())
		1.0
		>>> imgs = [np.zeros((1, 1, 3), dtype=np.float32)]
		>>> result = ImageAugmentor.normalize_images(imgs)
		>>> float(result.min())
		0.0
		"""
		all_images_array = np.asarray(images, dtype=np.float32)
		normalized_images = all_images_array / 255.0
		return normalized_images

	@staticmethod
	def split_dataset(
		images: np.ndarray,
		labels: np.ndarray,
		validation_size: float = 0.2,
		validation_to_test_ratio: float = 0.5,
		random_state: int = 42,
		label_offset: int = 1,
	) -> DatasetSplits:
		"""Split data into train, validation, and test sets."""
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

	def augment_and_split(
		self,
		images: List[np.ndarray],
		labels: List[int],
		validation_size: float = 0.2,
		validation_to_test_ratio: float = 0.5,
		random_state: int = 42,
		label_offset: int = 1,
	) -> DatasetSplits:
		"""Full pipeline to augment, combine, normalize, and split dataset."""
		augmented_images, augmented_labels = self.augment_dataset(images, labels)
		all_images, all_labels = self.combine_images(images, labels, augmented_images, augmented_labels)
		all_images_normalized = self.normalize_images(all_images)
		return self.split_dataset(all_images_normalized, all_labels, validation_size, validation_to_test_ratio, random_state, label_offset)
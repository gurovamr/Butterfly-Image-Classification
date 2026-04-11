"""Model training utilities with callbacks and history tracking."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential


class ModelTrainer:
	"""Handles model training with callbacks and history tracking.
	
	Provides configurable training callbacks including early stopping
	and learning rate reduction on plateau.
	"""

	def __init__(
		self,
		early_stopping_patience: int = 5,
		reduce_lr_patience: int = 3,
		reduce_lr_factor: float = 0.2,
		min_lr: float = 0.0001,
	):
		"""Initialize trainer with callback parameters.
		
		Parameters
		----------
		early_stopping_patience : int, default=5
			Number of epochs with no improvement to wait before stopping.
		reduce_lr_patience : int, default=3
			Number of epochs with no improvement before reducing learning rate.
		reduce_lr_factor : float, default=0.2
			Factor by which learning rate will be reduced (new_lr = lr * factor).
		min_lr : float, default=0.0001
			Minimum learning rate value.
		"""
		self.early_stopping_patience = early_stopping_patience
		self.reduce_lr_patience = reduce_lr_patience
		self.reduce_lr_factor = reduce_lr_factor
		self.min_lr = min_lr

	def create_callbacks(self) -> list[Callback]:
		"""Create training callbacks for early stopping and LR reduction.
		
		Returns
		-------
		list[Callback]
			List containing EarlyStopping and ReduceLROnPlateau callbacks.
		"""
		early_stopping = EarlyStopping(
			monitor="val_loss",
			patience=self.early_stopping_patience,
			restore_best_weights=True,
		)
		reduce_lr = ReduceLROnPlateau(
			monitor="val_loss",
			factor=self.reduce_lr_factor,
			patience=self.reduce_lr_patience,
			min_lr=self.min_lr,
		)
		return [early_stopping, reduce_lr]

	def train(
		self,
		model: Sequential,
		train_images: np.ndarray,
		train_labels: np.ndarray,
		val_images: np.ndarray,
		val_labels: np.ndarray,
		epochs: int = 30,
		callbacks: list[Callback] | None = None,
	) -> tf.keras.callbacks.History:
		"""Train the model on provided data.
		
		Parameters
		----------
		model : Sequential
			Compiled Keras model to train.
		train_images : np.ndarray
			Training images (normalized).
		train_labels : np.ndarray
			Training labels.
		val_images : np.ndarray
			Validation images (normalized).
		val_labels : np.ndarray
			Validation labels.
		epochs : int, default=30
			Number of training epochs.
		callbacks : list[Callback] | None, default=None
			Custom callbacks. If None, uses self.create_callbacks().
			
		Returns
		-------
		tf.keras.callbacks.History
			Keras History object containing training metrics.
		"""
		if callbacks is None:
			callbacks = self.create_callbacks()

		return model.fit(
			train_images,
			train_labels,
			epochs=epochs,
			validation_data=(val_images, val_labels),
			callbacks=callbacks,
		)
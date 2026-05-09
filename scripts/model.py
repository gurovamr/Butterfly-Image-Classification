"""CNN model architecture and persistence for butterfly classification."""

from pathlib import Path

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


class ButterflyClassifier:
    """CNN model for butterfly species classification.

    Provides two architecture variants: baseline and improved.
    The improved model includes additional batch normalization and dropout
    for better regularization.
    """

    def __init__(
        self,
        model_type: str = "improved",
        input_shape: tuple[int, int, int] = (128, 128, 3),
        num_classes: int = 10,
        learning_rate: float = 1e-4,
    ):
        """Initialize and build the butterfly classifier model.

        Parameters
        ----------
        model_type : str, default="improved"
                Type of model to build. Options: "baseline", "improved".
        input_shape : tuple[int, int, int], default=(128, 128, 3)
                Input image shape (height, width, channels).
        num_classes : int, default=10
                Number of butterfly species classes.
        learning_rate : float, default=1e-4
                Learning rate for Adam optimizer.
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model(model_type)

    def _build_model(self, model_type: str) -> Sequential:
        """Build model based on specified type."""
        if model_type == "baseline":
            return self._build_baseline()
        if model_type == "improved":
            return self._build_improved()
        raise ValueError(
            f"Unknown model type: {model_type}. " f"Choose from: 'baseline', 'improved'."
        )

    def _build_baseline(self) -> Sequential:
        """Build baseline CNN architecture.

        Architecture:
        - 3 Conv2D blocks with increasing filters
        - MaxPooling2D after each conv block
        - Single BatchNormalization after 2nd conv
        - Dense layers with decreasing units
        - No dropout

        Returns
        -------
        Sequential
                Compiled baseline model.
        """
        model = Sequential(
            [
                Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(256, activation="relu"),
                Dense(128, activation="relu"),
                Dense(self.num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _build_improved(self) -> Sequential:
        """Build improved CNN architecture with regularization.

        Architecture:
        - 3 Conv2D blocks with increasing filters
        - BatchNormalization after every conv block
        - MaxPooling2D after each conv block
        - Dense layers with decreasing units
        - Dropout in dense layers

        Returns
        -------
        Sequential
                Compiled improved model.
        """
        model = Sequential(
            [
                Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape),
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
                Dense(self.num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def save(self, output_path: str | Path) -> None:
        """Save the trained model to disk."""
        self.model.save(Path(output_path).as_posix())

    @staticmethod
    def load(model_path: str | Path) -> Sequential:
        """Load a saved Keras model from disk."""
        return load_model(Path(model_path).as_posix())

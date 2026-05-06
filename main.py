"""Project entry point for the butterfly classification workflow."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports
from pathlib import Path

import numpy as np

from scripts.config import DATA_DIR
from scripts.data_download import main as data_download
from scripts.dataset import (
    ImageAugmentor,
    ImageClassificationDataLoader,
)
from scripts.evaluator import ModelEvaluator
from scripts.model import ButterflyClassifier
from scripts.train import ModelTrainer
from scripts.visualizer import TrainingVisualizer

# pylint: enable=wrong-import-position,wrong-import-order,ungrouped-imports

PATH_IMAGES = "data/images"
PATH_MODELS = "models"
PATH_RESULTS = "results"


def run_workflow() -> None:
    """Run the full migrated workflow from data prep to evaluation."""
    # Setup paths
    data_download(DATA_DIR)
    images_dir = Path(PATH_IMAGES)
    model_output_dir = Path(PATH_MODELS)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(PATH_RESULTS)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize utilities
    visualizer = TrainingVisualizer(num_classes=10, save_dir=results_dir)
    trainer = ModelTrainer()
    evaluator = ModelEvaluator(num_classes=10)

    # Load and prepare data
    data_loader = ImageClassificationDataLoader(images_dir)
    original_images, original_labels = data_loader.load_images()
    print(f"There are total {len(original_images)} images in this dataset.")

    augmentor = ImageAugmentor(augmentations_per_image=4)
    splits = augmentor.augment_and_split(original_images, original_labels)

    all_images_normalized = np.concatenate(
        [splits.train_images, splits.val_images, splits.test_images],
        axis=0,
    )
    print(
        "Min pixel value after normalization: "
        f"{all_images_normalized.min()}, "
        f"Max pixel value after normalization: {all_images_normalized.max()}"
    )

    # Visualize data
    visualizer.plot_random_images(
        original_images, original_images, original_labels, original_labels, num_images=5
    )
    visualizer.plot_class_examples(original_images, original_labels, class_index=5, num_images=5)

    # Train baseline model
    print("\n=== Training Baseline Model ===")
    baseline_classifier = ButterflyClassifier(model_type="baseline")
    baseline_history = trainer.train(
        baseline_classifier.model,
        splits.train_images,
        splits.train_labels,
        splits.val_images,
        splits.val_labels,
        epochs=30,
    )
    visualizer.plot_training_history(baseline_history.history, title_suffix=" (Baseline)")
    _, baseline_test_accuracy = evaluator.evaluate(
        baseline_classifier.model,
        splits.test_images,
        splits.test_labels,
    )
    print(f"Baseline test accuracy: {baseline_test_accuracy:.4f}")
    baseline_classifier.save(model_output_dir / "augmented_model_normal.keras")

    # Train improved model
    print("\n=== Training Improved Model ===")
    improved_classifier = ButterflyClassifier(model_type="improved")
    improved_history = trainer.train(
        improved_classifier.model,
        splits.train_images,
        splits.train_labels,
        splits.val_images,
        splits.val_labels,
        epochs=30,
    )
    visualizer.plot_training_history(improved_history.history, title_suffix=" (Improved)")
    _, improved_test_accuracy = evaluator.evaluate(
        improved_classifier.model,
        splits.test_images,
        splits.test_labels,
    )
    print(f"Improved test accuracy: {improved_test_accuracy:.4f}")
    improved_classifier.save(model_output_dir / "augmented_model_upd.keras")

    # Evaluate and visualize predictions
    print("\n=== Analyzing Predictions ===")
    test_predicted_labels = evaluator.predict(improved_classifier.model, splits.test_images)
    visualizer.plot_confusion_matrix(splits.test_labels, test_predicted_labels)
    visualizer.plot_prediction_samples(
        splits.test_images,
        splits.test_labels,
        test_predicted_labels,
        num_images=10,
    )
    visualizer.plot_class_histogram(splits.test_labels, test_predicted_labels)

    # Calculate per-class accuracy
    accuracy_per_class = evaluator.calculate_class_accuracy(
        splits.test_labels, test_predicted_labels
    )
    for class_label, accuracy in accuracy_per_class.items():
        print(f"Class {class_label}: {accuracy * 100:.2f}%")
    least_accurate_class = min(accuracy_per_class, key=accuracy_per_class.get)
    print(
        "Class with the least accuracy: "
        f"{least_accurate_class} ({accuracy_per_class[least_accurate_class] * 100:.2f}%)"
    )

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Models saved to:      {model_output_dir.absolute()}")
    print(f"Plots saved to:       {results_dir.absolute()}")
    print(f"Baseline accuracy:    {baseline_test_accuracy:.4f}")
    print(f"Improved accuracy:    {improved_test_accuracy:.4f}")
    print("=" * 60)


def main() -> None:
    """Program entry point."""
    run_workflow()


if __name__ == "__main__":
    main()

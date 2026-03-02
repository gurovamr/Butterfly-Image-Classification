"""Project entry point for the butterfly classification workflow."""

from __future__ import annotations

from pathlib import Path

from scripts.data_loader import (
	combine_and_normalize_images,
	create_augmented_dataset,
	load_labeled_images_from_directory,
	split_dataset,
)
from scripts.plotting import (
	plot_class_examples,
	plot_class_prediction_histogram,
	plot_confusion_matrix_from_labels,
	plot_random_images,
	plot_random_prediction_samples,
	plot_training_history,
)
from scripts.processing import (
	build_baseline_model,
	build_improved_model,
	calculate_class_accuracy,
	create_training_callbacks,
	evaluate_model,
	predict_class_labels,
	save_model_artifact,
	train_model,
)


def run_workflow() -> None:
	"""Run the full migrated workflow from data prep to evaluation."""
	images_dir = Path("data/images")
	model_output_dir = Path("models")
	model_output_dir.mkdir(parents=True, exist_ok=True)

	original_images, original_labels = load_labeled_images_from_directory(images_dir)
	print(f"There are total {len(original_images)} images in this dataset.")

	augmented_images, augmented_labels = create_augmented_dataset(
		original_images,
		original_labels,
		augmentations_per_image=4,
	)

	all_images_normalized, all_labels = combine_and_normalize_images(
		original_images,
		original_labels,
		augmented_images,
		augmented_labels,
	)
	print(
		"Min pixel value after normalization: "
		f"{all_images_normalized.min()}, Max pixel value after normalization: {all_images_normalized.max()}"
	)

	plot_random_images(original_images, augmented_images, original_labels, augmented_labels, num_images=5)
	plot_class_examples(original_images, original_labels, class_index=5, num_images=5)

	splits = split_dataset(all_images_normalized, all_labels)
	callbacks = create_training_callbacks()

	baseline_model = build_baseline_model()
	baseline_history = train_model(
		baseline_model,
		splits.train_images,
		splits.train_labels,
		splits.val_images,
		splits.val_labels,
		epochs=30,
		callbacks=callbacks,
	)
	plot_training_history(baseline_history.history, title_suffix=" (Baseline)")
	_, baseline_test_accuracy = evaluate_model(
		baseline_model,
		splits.test_images,
		splits.test_labels,
	)
	print(f"Baseline test accuracy: {baseline_test_accuracy:.4f}")
	save_model_artifact(baseline_model, model_output_dir / "augmented_model_normal.keras")

	improved_model = build_improved_model()
	improved_history = train_model(
		improved_model,
		splits.train_images,
		splits.train_labels,
		splits.val_images,
		splits.val_labels,
		epochs=30,
		callbacks=callbacks,
	)
	plot_training_history(improved_history.history, title_suffix=" (Improved)")
	_, improved_test_accuracy = evaluate_model(
		improved_model,
		splits.test_images,
		splits.test_labels,
	)
	print(f"Improved test accuracy: {improved_test_accuracy:.4f}")
	save_model_artifact(improved_model, model_output_dir / "augmented_model_upd.keras")

	test_predicted_labels = predict_class_labels(improved_model, splits.test_images)
	plot_confusion_matrix_from_labels(splits.test_labels, test_predicted_labels)
	plot_random_prediction_samples(
		splits.test_images,
		splits.test_labels,
		test_predicted_labels,
		num_images=10,
	)
	plot_class_prediction_histogram(splits.test_labels, test_predicted_labels)

	accuracy_per_class = calculate_class_accuracy(splits.test_labels, test_predicted_labels, num_classes=10)
	for class_label, accuracy in accuracy_per_class.items():
		print(f"Class {class_label}: {accuracy * 100:.2f}%")
	least_accurate_class = min(accuracy_per_class, key=accuracy_per_class.get)
	print(
		"Class with the least accuracy: "
		f"{least_accurate_class} ({accuracy_per_class[least_accurate_class] * 100:.2f}%)"
	)


def main() -> None:
	"""Program entry point."""
	run_workflow()


if __name__ == "__main__":
	main()
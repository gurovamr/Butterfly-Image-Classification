import pytest
import numpy as np
import matplotlib.image as mpimg

from scripts.visualizer import TrainingVisualizer


class TestPlotRandomImages:
    # Equivalence class 1: normal call — output file is created on disk with valid content
    def test_creates_output_file(self, tmp_path, make_image):
        viz = TrainingVisualizer(save_dir=tmp_path)
        images = [make_image(h=32, w=32)] * 6
        labels = [1] * 6
        viz.plot_random_images(images, images, labels, labels, num_images=2)
        output = tmp_path / "01_random_images_comparison.png"
        assert output.exists()
        img = mpimg.imread(str(output))
        assert img.ndim == 3 and img.shape[0] > 0 and img.shape[1] > 0


class TestPlotClassExamples:
    # Equivalence class 1: valid class index — output file is created with valid content
    def test_creates_output_file_for_valid_index(self, tmp_path, make_image):
        viz = TrainingVisualizer(save_dir=tmp_path)
        images = [make_image(h=32, w=32)] * 4
        labels = [1, 1, 2, 2]
        viz.plot_class_examples(images, labels, class_index=0, num_images=2)
        output = tmp_path / "02_class_1_examples.png"
        assert output.exists()
        img = mpimg.imread(str(output))
        assert img.ndim == 3 and img.shape[0] > 0 and img.shape[1] > 0

    # Equivalence class 2: edge case — class_index too high raises ValueError
    def test_raises_value_error_for_out_of_range_index(self, tmp_path, make_image):
        viz = TrainingVisualizer(save_dir=tmp_path)
        images = [make_image(h=32, w=32)] * 3
        labels = [1, 1, 2]
        with pytest.raises(ValueError, match="class_index is out of range"):
            viz.plot_class_examples(images, labels, class_index=99)

    # Equivalence class 3: edge case — negative class_index raises ValueError
    def test_raises_value_error_for_negative_index(self, tmp_path, make_image):
        viz = TrainingVisualizer(save_dir=tmp_path)
        images = [make_image(h=32, w=32)] * 2
        labels = [1, 1]
        with pytest.raises(ValueError, match="class_index is out of range"):
            viz.plot_class_examples(images, labels, class_index=-1)


class TestPlotTrainingHistory:
    # Equivalence class 1: valid history dict — both loss and accuracy files are created with valid content
    def test_creates_loss_and_accuracy_files(self, tmp_path):
        viz = TrainingVisualizer(save_dir=tmp_path)
        history = {
            "loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "accuracy": [0.7, 0.8, 0.9],
            "val_accuracy": [0.65, 0.75, 0.85],
        }
        viz.plot_training_history(history, title_suffix=" (Test)")
        for filename in ("03_loss_test.png", "04_accuracy_test.png"):
            output = tmp_path / filename
            assert output.exists()
            img = mpimg.imread(str(output))
            assert img.ndim == 3 and img.shape[0] > 0 and img.shape[1] > 0

    # Equivalence class 2: edge case — empty history dict does not raise
    def test_empty_history_does_not_raise(self, tmp_path):
        viz = TrainingVisualizer(save_dir=tmp_path)
        viz.plot_training_history({})
        assert (tmp_path / "03_loss.png").exists()
        assert (tmp_path / "04_accuracy.png").exists()


class TestPlotConfusionMatrix:
    # Equivalence class 1: normal call — output file is created on disk with valid content
    def test_creates_output_file(self, tmp_path):
        viz = TrainingVisualizer(num_classes=3, save_dir=tmp_path)
        true_labels = np.array([0, 1, 2, 0, 1])
        predicted_labels = np.array([0, 2, 2, 1, 1])
        viz.plot_confusion_matrix(true_labels, predicted_labels)
        output = tmp_path / "05_confusion_matrix.png"
        assert output.exists()
        img = mpimg.imread(str(output))
        assert img.ndim == 3 and img.shape[0] > 0 and img.shape[1] > 0

    # Equivalence class 2: perfect predictions — diagonal-only confusion matrix, file created
    def test_perfect_predictions_creates_output_file(self, tmp_path):
        viz = TrainingVisualizer(num_classes=3, save_dir=tmp_path)
        labels = np.array([0, 1, 2])
        viz.plot_confusion_matrix(labels, labels)
        assert (tmp_path / "05_confusion_matrix.png").exists()


class TestPlotPredictionSamples:
    # Equivalence class 1: normal call — output file is created on disk with valid content
    def test_creates_output_file(self, tmp_path, make_image):
        viz = TrainingVisualizer(save_dir=tmp_path)
        images = np.stack([make_image(h=32, w=32)] * 6)
        true_labels = np.array([0, 1, 2, 0, 1, 2])
        predicted_labels = np.array([0, 1, 0, 0, 2, 2])
        viz.plot_prediction_samples(images, true_labels, predicted_labels, num_images=4)
        output = tmp_path / "06_prediction_samples.png"
        assert output.exists()
        img = mpimg.imread(str(output))
        assert img.ndim == 3 and img.shape[0] > 0 and img.shape[1] > 0

    # Equivalence class 2: edge case — num_images larger than dataset is capped to len(images)
    def test_num_images_capped_to_dataset_size(self, tmp_path, make_image):
        viz = TrainingVisualizer(save_dir=tmp_path)
        images = np.stack([make_image(h=32, w=32)] * 3)
        labels = np.array([0, 1, 2])
        viz.plot_prediction_samples(images, labels, labels, num_images=100)
        assert (tmp_path / "06_prediction_samples.png").exists()


class TestPlotClassHistogram:
    # Equivalence class 1: mixed correct/incorrect predictions — output file is created with valid content
    def test_creates_output_file(self, tmp_path):
        viz = TrainingVisualizer(num_classes=3, save_dir=tmp_path)
        true_labels = np.array([0, 0, 1, 1, 2, 2])
        predicted_labels = np.array([0, 1, 1, 0, 2, 2])
        viz.plot_class_histogram(true_labels, predicted_labels)
        output = tmp_path / "07_class_prediction_histogram.png"
        assert output.exists()
        img = mpimg.imread(str(output))
        assert img.ndim == 3 and img.shape[0] > 0 and img.shape[1] > 0

    # Equivalence class 2: all predictions correct — correct counts equal total counts
    def test_all_correct_creates_output_file(self, tmp_path):
        viz = TrainingVisualizer(num_classes=3, save_dir=tmp_path)
        labels = np.array([0, 1, 2])
        viz.plot_class_histogram(labels, labels)
        assert (tmp_path / "07_class_prediction_histogram.png").exists()

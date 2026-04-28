import pytest
import numpy as np

from scripts.visualizer import TrainingVisualizer


class TestPlotRandomImages:
    # Equivalence class 1: normal call — output file is created on disk
    def test_creates_output_file(self, tmp_path, make_image):
        viz = TrainingVisualizer(save_dir=tmp_path)
        images = [make_image(h=32, w=32)] * 6
        labels = [1] * 6
        viz.plot_random_images(images, images, labels, labels, num_images=2)
        assert (tmp_path / "01_random_images_comparison.png").exists()


class TestPlotClassExamples:
    # Equivalence class 1: valid class index — output file is created
    def test_creates_output_file_for_valid_index(self, tmp_path, make_image):
        viz = TrainingVisualizer(save_dir=tmp_path)
        images = [make_image(h=32, w=32)] * 4
        labels = [1, 1, 2, 2]
        viz.plot_class_examples(images, labels, class_index=0, num_images=2)
        assert (tmp_path / "02_class_1_examples.png").exists()

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
    # Equivalence class 1: valid history dict — both loss and accuracy files are created
    def test_creates_loss_and_accuracy_files(self, tmp_path):
        viz = TrainingVisualizer(save_dir=tmp_path)
        history = {
            "loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "accuracy": [0.7, 0.8, 0.9],
            "val_accuracy": [0.65, 0.75, 0.85],
        }
        viz.plot_training_history(history, title_suffix=" (Test)")
        assert (tmp_path / "03_loss_test.png").exists()
        assert (tmp_path / "04_accuracy_test.png").exists()

    # Equivalence class 2: edge case — empty history dict does not raise
    def test_empty_history_does_not_raise(self, tmp_path):
        viz = TrainingVisualizer(save_dir=tmp_path)
        viz.plot_training_history({})
        assert (tmp_path / "03_loss.png").exists()
        assert (tmp_path / "04_accuracy.png").exists()

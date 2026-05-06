import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from scripts.dataset import DatasetSplits, ImageAugmentor, ImageClassificationDataLoader


class TestListImageFilePaths:
    # Equivalence class 1: directory with image files only
    def test_returns_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "img1.jpg").touch()
            (d / "img2.png").touch()
            loader = ImageClassificationDataLoader(d)
            assert len(loader._list_image_file_paths()) == 2

    # Equivalence class 2: directory contains a subdirectory — subdirs are ignored
    def test_ignores_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "img1.jpg").touch()
            (d / "subdir").mkdir()
            loader = ImageClassificationDataLoader(d)
            assert len(loader._list_image_file_paths()) == 1

    # Equivalence class 3: edge case — empty directory returns empty list
    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ImageClassificationDataLoader(tmpdir)
            assert loader._list_image_file_paths() == []


class TestExtractLabels:
    # Equivalence class 1: valid filename format — label parsed correctly
    def test_parses_label_from_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            paths = [d / "0010001.png", d / "0050002.png"]
            loader = ImageClassificationDataLoader(d)
            result = loader._extract_labels(paths)
            assert result[paths[0]] == 1
            assert result[paths[1]] == 5

    # Equivalence class 2: mixed valid and invalid filenames — invalid ones skipped
    def test_mixed_valid_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            paths = [d / "0030001.png", d / "bad.png"]
            loader = ImageClassificationDataLoader(d)
            result = loader._extract_labels(paths)
            assert len(result) == 1
            assert result[paths[0]] == 3


class TestLoadImages:
    # Equivalence class 1: valid path-to-label map — returns correct images and labels
    def test_returns_images_and_labels(self, mocker, make_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            fake_img = make_image()
            path_to_label = {d / "0010001.png": 1, d / "0020002.png": 2}
            loader = ImageClassificationDataLoader(d)
            mocker.patch("scripts.dataset.load_img", return_value=MagicMock())
            mocker.patch("scripts.dataset.img_to_array", return_value=fake_img)
            images, labels = loader._load_images(path_to_label)
        assert len(images) == 2
        assert labels == [1, 2]


class TestAugmentSingle:
    # Equivalence class 1: output dtype is float32
    def test_output_dtype_is_float32(self, make_image):
        augmentor = ImageAugmentor(augmentations_per_image=1)
        img = make_image()
        imgs, _ = augmentor._augment_single(img, label=2)
        assert imgs[0].dtype == np.float32

    # Equivalence class 2: output spatial shape matches input shape
    def test_output_shape_preserved(self, make_image):
        augmentor = ImageAugmentor(augmentations_per_image=2)
        img = make_image(h=128, w=128)
        imgs, _ = augmentor._augment_single(img, label=1)
        assert imgs[0].shape == (128, 128, 3)


class TestAugmentDataset:
    # Equivalence class 1: output length equals n_images * augmentations_per_image
    def test_length_equals_n_images_times_augmentations(self, make_image):
        augmentor = ImageAugmentor(augmentations_per_image=2)
        images = [make_image(), make_image()]
        labels = [1, 2]
        aug_imgs, aug_lbls = augmentor.augment_dataset(images, labels)
        assert len(aug_imgs) == 4
        assert len(aug_lbls) == 4


class TestCombineImages:
    # Equivalence class 1: both lists non-empty — concatenated correctly
    def test_concatenates_both_lists(self, make_image):
        orig = [make_image()]
        aug = [make_image(), make_image()]
        imgs, lbls = ImageAugmentor.combine_images(orig, [1], aug, [2, 3])
        assert len(imgs) == 3
        assert lbls == [1, 2, 3]

    # Equivalence class 2: edge case — empty augmented list returns original only
    def test_empty_augmented_returns_original_only(self, make_image):
        orig = [make_image()]
        imgs, lbls = ImageAugmentor.combine_images(orig, [1], [], [])
        assert len(imgs) == 1
        assert lbls == [1]


class TestNormalizeImages:
    # Equivalence class 1: max pixel value (255) maps to 1.0
    def test_scales_max_to_1(self, make_image):
        imgs = [make_image(fill=255.0)]
        result = ImageAugmentor.normalize_images(imgs)
        assert result.max() == pytest.approx(1.0)

    # Equivalence class 2: edge case — zero pixel value maps to 0.0
    def test_scales_min_to_0(self, make_image):
        imgs = [make_image(fill=0.0)]
        result = ImageAugmentor.normalize_images(imgs)
        assert result.min() == pytest.approx(0.0)

    # Equivalence class 3: output dtype is always float32
    def test_output_is_float32(self, make_image):
        imgs = [make_image()]
        result = ImageAugmentor.normalize_images(imgs)
        assert result.dtype == np.float32


class TestSplitDataset:
    # Equivalence class 1: output is a DatasetSplits instance
    def test_returns_dataset_splits(self):
        images = np.zeros((20, 128, 128, 3), dtype=np.float32)
        labels = np.array([i % 5 + 1 for i in range(20)], dtype=np.int32)
        splits = ImageAugmentor.split_dataset(images, labels)
        assert isinstance(splits, DatasetSplits)

    # Equivalence class 2: total samples across all splits equals input size
    def test_total_samples_preserved(self):
        n = 20
        images = np.zeros((n, 128, 128, 3), dtype=np.float32)
        labels = np.array([i % 5 + 1 for i in range(n)], dtype=np.int32)
        splits = ImageAugmentor.split_dataset(images, labels)
        total = len(splits.train_images) + len(splits.val_images) + len(splits.test_images)
        assert total == n

    # Equivalence class 3: label offset is applied — raw labels starting at 1 become 0-indexed
    def test_label_offset_applied(self):
        images = np.zeros((20, 4, 4, 3), dtype=np.float32)
        labels = np.ones(20, dtype=np.int32)  # all label=1, offset=1 → should become 0
        splits = ImageAugmentor.split_dataset(images, labels, label_offset=1)
        assert splits.train_labels.min() == 0
        assert splits.train_labels.max() == 0


class TestLoadImagesPipeline:
    # Equivalence class 1: full pipeline — load_images() calls internal steps and returns images+labels
    def test_load_images_pipeline(self, mocker, make_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            fake_img = make_image()
            (d / "0010001.png").touch()
            loader = ImageClassificationDataLoader(d)
            mocker.patch("scripts.dataset.load_img", return_value=MagicMock())
            mocker.patch("scripts.dataset.img_to_array", return_value=fake_img)
            images, labels = loader.load_images()
        assert len(images) == 1
        assert labels == [1]


class TestPrepareDataset:
    # Equivalence class 1: full pipeline — augment_and_split() returns a DatasetSplits instance
    def test_augment_and_split_returns_splits(self, make_image):
        augmentor = ImageAugmentor(augmentations_per_image=1)
        images = [make_image() for _ in range(20)]
        labels = [i % 5 + 1 for i in range(20)]
        splits = augmentor.augment_and_split(images, labels)
        assert isinstance(splits, DatasetSplits)
        total = len(splits.train_images) + len(splits.val_images) + len(splits.test_images)
        assert total == 40  # 20 original + 20 augmented (1 per image)

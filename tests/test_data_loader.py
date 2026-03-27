import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.data_loader import DatasetSplits, ImageAugmentor, ImageClassificationDataLoader


def _make_image(h: int = 128, w: int = 128, c: int = 3, fill: float = 128.0) -> np.ndarray:
    return np.full((h, w, c), fill, dtype=np.float32)



class TestListImageFilePaths:
    def test_returns_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "img1.jpg").touch()
            (d / "img2.png").touch()
            loader = ImageClassificationDataLoader(d)
            assert len(loader._list_image_file_paths()) == 2

    def test_ignores_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "img1.jpg").touch()
            (d / "subdir").mkdir()
            loader = ImageClassificationDataLoader(d)
            assert len(loader._list_image_file_paths()) == 1

    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ImageClassificationDataLoader(tmpdir)
            assert loader._list_image_file_paths() == []


class TestExtractLabels:
    def test_parses_label_from_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            paths = [d / "0010001.png", d / "0050002.png"]
            loader = ImageClassificationDataLoader(d)
            result = loader._extract_labels(paths)
            assert result[paths[0]] == 1
            assert result[paths[1]] == 5

 

    def test_mixed_valid_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            paths = [d / "0030001.png", d / "bad.png"]
            loader = ImageClassificationDataLoader(d)
            result = loader._extract_labels(paths)
            assert len(result) == 1
            assert result[paths[0]] == 3


class TestLoadImages:
    def test_returns_images_and_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            fake_img = _make_image()
            path_to_label = {d / "0010001.png": 1, d / "0020002.png": 2}
            loader = ImageClassificationDataLoader(d)
            with patch("scripts.data_loader.load_img") as mock_load, \
                 patch("scripts.data_loader.img_to_array", return_value=fake_img):
                mock_load.return_value = MagicMock()
                images, labels = loader._load_images(path_to_label)
            assert len(images) == 2
            assert labels == [1, 2]

class TestAugmentSingle:
    def test_output_dtype_is_float32(self):
        augmentor = ImageAugmentor(augmentations_per_image=1)
        img = _make_image()
        imgs, _ = augmentor._augment_single(img, label=2)
        assert imgs[0].dtype == np.float32

    def test_output_shape_preserved(self):
        augmentor = ImageAugmentor(augmentations_per_image=2)
        img = _make_image(h=128, w=128)
        imgs, _ = augmentor._augment_single(img, label=1)
        assert imgs[0].shape == (128, 128, 3)


class TestAugmentDataset:
    def test_length_equals_n_images_times_augmentations(self):
        augmentor = ImageAugmentor(augmentations_per_image=2)
        images = [_make_image(), _make_image()]
        labels = [1, 2]
        aug_imgs, aug_lbls = augmentor.augment_dataset(images, labels)
        assert len(aug_imgs) == 4
        assert len(aug_lbls) == 4

class TestCombineImages:
    def test_concatenates_both_lists(self):
        orig = [_make_image()]
        aug = [_make_image(), _make_image()]
        imgs, lbls = ImageAugmentor.combine__images(orig, [1], aug, [2, 3])
        assert len(imgs) == 3
        assert lbls == [1, 2, 3]

    def test_empty_augmented_returns_original_only(self):
        orig = [_make_image()]
        imgs, lbls = ImageAugmentor.combine__images(orig, [1], [], [])
        assert len(imgs) == 1
        assert lbls == [1]


class TestNormalizeImages:
    def test_scales_max_to_1(self):
        imgs = [_make_image(fill=255.0)]
        result = ImageAugmentor.normalize_images(imgs)
        assert result.max() == pytest.approx(1.0)

    def test_scales_min_to_0(self):
        imgs = [_make_image(fill=0.0)]
        result = ImageAugmentor.normalize_images(imgs)
        assert result.min() == pytest.approx(0.0)

    def test_output_is_float32(self):
        imgs = [_make_image()]
        result = ImageAugmentor.normalize_images(imgs)
        assert result.dtype == np.float32


class TestSplitDataset:
    def test_returns_dataset_splits(self):
        images = np.random.rand(20, 128, 128, 3).astype(np.float32)
        labels = np.array([i % 5 + 1 for i in range(20)], dtype=np.int32)
        splits = ImageAugmentor.split_dataset(images, labels)
        assert isinstance(splits, DatasetSplits)

    def test_total_samples_preserved(self):
        n = 20
        images = np.random.rand(n, 128, 128, 3).astype(np.float32)
        labels = np.array([i % 5 + 1 for i in range(n)], dtype=np.int32)
        splits = ImageAugmentor.split_dataset(images, labels)
        total = len(splits.train_images) + len(splits.val_images) + len(splits.test_images)
        assert total == n



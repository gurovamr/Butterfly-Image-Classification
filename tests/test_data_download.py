import pytest
import zipfile
import tempfile
from pathlib import Path

from scripts.data_download import (
    check_if_dataset_exists,
    create_data_dir,
    show_progress,
    flatten_data_dir,
    download_data,
    extract_data,
    cleanup,
    main,
)


class TestCheckIfDatasetExists:
    # Equivalence class 1: images directory exists and contains files
    def test_dataset_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            images_dir = data_dir / "images"
            images_dir.mkdir()
            (images_dir / "img.jpg").touch()
            assert check_if_dataset_exists(data_dir) is True

    # Equivalence class 2: images directory exists but is empty
    def test_dataset_not_exists_when_images_dir_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "images").mkdir()
            assert check_if_dataset_exists(data_dir) is False

    # Equivalence class 3: data directory does not exist at all
    def test_dataset_does_not_exist(self):
        data_dir = Path("/nonexistent/path/to/data")
        assert check_if_dataset_exists(data_dir) is False


class TestCreateDataDir:
    # Equivalence class 1: basic single-level creation
    def test_create_single_level_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            create_data_dir(data_dir)
            assert data_dir.exists()

    # Equivalence class 2: nested (multi-level) directory creation
    def test_create_nested_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "a" / "b" / "c"
            create_data_dir(data_dir)
            assert data_dir.exists()

    # Equivalence class 3: edge case — directory already exists (idempotent)
    def test_idempotent_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            create_data_dir(data_dir)
            create_data_dir(data_dir)
            assert data_dir.exists()


class TestShowProgress:
    # Equivalence class 1: normal progress with known total size
    def test_show_progress_output(self, capsys):
        show_progress(50, 1024, 512 * 1024)
        captured = capsys.readouterr()
        assert "Downloading:" in captured.out
        assert "%" in captured.out

    # Equivalence class 2: edge case — total size is zero (no division by zero)
    def test_show_progress_zero_total_size(self, capsys):
        show_progress(0, 1024, 0)
        captured = capsys.readouterr()
        assert "0.00%" in captured.out


class TestFlattenDataDir:
    # Equivalence class 1: exactly one nested directory — should be flattened
    def test_flatten_single_nested_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            nested_dir = data_dir / "nested"
            nested_dir.mkdir()
            (nested_dir / "file.txt").touch()

            flatten_data_dir(data_dir)

            assert (data_dir / "file.txt").exists()
            assert not nested_dir.exists()

    # Equivalence class 2: multiple nested directories — should not flatten
    def test_no_flatten_multiple_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            dir1 = data_dir / "dir1"
            dir2 = data_dir / "dir2"
            dir1.mkdir()
            dir2.mkdir()

            flatten_data_dir(data_dir)

            assert dir1.exists()
            assert dir2.exists()

    # Equivalence class 3: edge case — empty directory, nothing to do
    def test_no_flatten_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            flatten_data_dir(data_dir)
            assert data_dir.exists()

    # Equivalence class 4: edge case — file collision: target already exists — nested file is skipped
    def test_skips_file_when_target_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            nested_dir = data_dir / "nested"
            nested_dir.mkdir()
            (nested_dir / "file.txt").write_text("nested content")
            (data_dir / "file.txt").write_text("existing content")  # collision

            flatten_data_dir(data_dir)

            # Existing file must not be overwritten
            assert (data_dir / "file.txt").read_text() == "existing content"


class TestDownloadData:
    # Equivalence class 1: successful download — correct path returned and urlretrieve called
    def test_download_data_calls_urlretrieve_and_returns_zip_path(self, mocker):
        mock_urlretrieve = mocker.patch("scripts.data_download.urllib.request.urlretrieve")
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            url = "https://example.com/dataset.zip"

            result = download_data(url, data_dir)

            expected_zip_path = data_dir / "leedsbutterfly_dataset.zip"
            assert result == expected_zip_path
            mock_urlretrieve.assert_called_once_with(url, expected_zip_path, show_progress)


class TestExtractData:
    # Equivalence class 1: valid zip file is extracted correctly
    def test_extract_zip_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "test.zip"
            extract_dir = Path(tmpdir) / "extracted"
            extract_dir.mkdir()

            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("file.txt", "content")

            extract_data(zip_path, extract_dir)

            assert (extract_dir / "file.txt").exists()


class TestCleanup:
    # Equivalence class 1: file exists and is removed
    def test_cleanup_removes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "test.zip"
            zip_path.touch()

            cleanup(zip_path)

            assert not zip_path.exists()


class TestMain:
    # Equivalence class 1: dataset already present — download is skipped entirely
    def test_skips_download_if_dataset_exists(self, mocker):
        mocker.patch("scripts.data_download.check_if_dataset_exists", return_value=True)
        mock_download = mocker.patch("scripts.data_download.download_data")

        main(Path("/some/dir"))

        mock_download.assert_not_called()

    # Equivalence class 2: dataset missing — download, extract, and cleanup all called
    def test_downloads_and_extracts_if_missing(self, mocker, tmp_path):
        mocker.patch("scripts.data_download.check_if_dataset_exists", return_value=False)
        mocker.patch("scripts.data_download.create_data_dir")
        mock_download = mocker.patch(
            "scripts.data_download.download_data",
            return_value=tmp_path / "data.zip",
        )
        mock_extract = mocker.patch("scripts.data_download.extract_data")
        mock_cleanup = mocker.patch("scripts.data_download.cleanup")

        main(tmp_path)

        mock_download.assert_called_once()
        mock_extract.assert_called_once()
        mock_cleanup.assert_called_once()

    # Equivalence class 3: edge case — cleanup runs even if extract raises an exception
    def test_cleanup_runs_even_if_extract_raises(self, mocker, tmp_path):
        mocker.patch("scripts.data_download.check_if_dataset_exists", return_value=False)
        mocker.patch("scripts.data_download.create_data_dir")
        mocker.patch(
            "scripts.data_download.download_data",
            return_value=tmp_path / "data.zip",
        )
        mocker.patch(
            "scripts.data_download.extract_data",
            side_effect=Exception("extract failed"),
        )
        mock_cleanup = mocker.patch("scripts.data_download.cleanup")

        with pytest.raises(Exception):
            main(tmp_path)

        mock_cleanup.assert_called_once()

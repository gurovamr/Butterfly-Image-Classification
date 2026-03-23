import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import zipfile
import tempfile

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
    def test_dataset_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            assert check_if_dataset_exists(data_dir) is True

    def test_dataset_does_not_exist(self):
        data_dir = Path("/nonexistent/path/to/data")
        assert check_if_dataset_exists(data_dir) is False


class TestCreateDataDir:
    def test_create_single_level_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            create_data_dir(data_dir)
            assert data_dir.exists()

    def test_create_nested_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "a" / "b" / "c"
            create_data_dir(data_dir)
            assert data_dir.exists()

    def test_idempotent_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            create_data_dir(data_dir)
            create_data_dir(data_dir)
            assert data_dir.exists()


class TestShowProgress:
    def test_show_progress_output(self, capsys):
        show_progress(50, 1024, 512 * 1024)
        captured = capsys.readouterr()
        assert "Downloading:" in captured.out
        assert "%" in captured.out

    def test_show_progress_zero_total_size(self, capsys):
        show_progress(0, 1024, 0)
        captured = capsys.readouterr()
        assert "0.00%" in captured.out


class TestFlattenDataDir:
    def test_flatten_single_nested_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            nested_dir = data_dir / "nested"
            nested_dir.mkdir()
            (nested_dir / "file.txt").touch()
            
            flatten_data_dir(data_dir)
            
            assert (data_dir / "file.txt").exists()
            assert not nested_dir.exists()

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

    def test_no_flatten_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            flatten_data_dir(data_dir)
            assert data_dir.exists()

class TestDownloadData:
    @patch("scripts.data_download.urllib.request.urlretrieve")
    def test_download_data_calls_urlretrieve_and_returns_zip_path(self, mock_urlretrieve):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            url = "https://example.com/dataset.zip"

            result = download_data(url, data_dir)

            expected_zip_path = data_dir / "leedsbutterfly_dataset.zip"
            assert result == expected_zip_path
            mock_urlretrieve.assert_called_once_with(url, expected_zip_path, show_progress)

class TestExtractData:
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
    def test_cleanup_removes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "test.zip"
            zip_path.touch()
            
            cleanup(zip_path)
            
            assert not zip_path.exists()


class TestMain:
    @patch("scripts.data_download.download_data")
    @patch("scripts.data_download.extract_data")
    def test_main_dataset_exists(self, mock_extract, mock_download):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            main(data_dir)
            
            mock_download.assert_not_called()
            mock_extract.assert_not_called()

    @patch("scripts.data_download.cleanup")
    @patch("scripts.data_download.extract_data")
    @patch("scripts.data_download.download_data")
    def test_main_dataset_missing(self, mock_download, mock_extract, mock_cleanup):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "new_data"
            mock_download.return_value = Path(tmpdir) / "test.zip"
            
            main(data_dir)
            
            assert data_dir.exists()
            mock_download.assert_called_once()
            mock_extract.assert_called_once()
            mock_cleanup.assert_called_once()
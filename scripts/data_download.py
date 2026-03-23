import zipfile
from pathlib import Path
import urllib.request

ZENODO_URL = "https://zenodo.org/records/7559420/files/leedsbutterfly_dataset_v1.1.zip?download=1"

def check_if_dataset_exists(data_dir: Path) -> bool:
    """Check whether `data_dir` exists"""
    return data_dir.exists()

def create_data_dir(data_dir: Path) -> None:
    """Create the data directory if it does not exist."""
    data_dir.mkdir(parents=True, exist_ok=True)

def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded / total_size * 100, 100)
    else:
        percent = 0.0
    print(f"\rDownloading: {percent:.2f}%", end="")

def flatten_data_dir(data_dir: Path) -> None:
    """Flatten nested root folder."""
    nested_directories = [directory for directory in data_dir.iterdir() if directory.is_dir()]
    if len(nested_directories) != 1:
        return
    
    nested_root_dir = nested_directories[0]
    for item in list(nested_root_dir.iterdir()):
        target = data_dir / item.name
        if target.exists():
            continue
        item.rename(target)

    if not any(nested_root_dir.iterdir()):
        nested_root_dir.rmdir()

def download_data(url: str, data_dir: Path) -> Path:
    """Download the dataset ZIP into `data_dir` and return its path."""
    zip_path = data_dir / "leedsbutterfly_dataset.zip"
    urllib.request.urlretrieve(url, zip_path, show_progress)
    print()
    return zip_path

def extract_data(zip_path: Path, extract_to: Path) -> None:
    """Extract the downloaded ZIP file to `extract_to`."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    flatten_data_dir(extract_to)

def cleanup(zip_path: Path) -> None:
    """Remove the downloaded ZIP file after extraction."""
    zip_path.unlink()

def main(data_dir: Path) -> None:
    """Check whether the dataset is present; download and extract it into `data_dir` if missing."""
    if check_if_dataset_exists(data_dir):
        return

    create_data_dir(data_dir)
    zip_path = download_data(ZENODO_URL, data_dir)

    try:
        extract_data(zip_path, data_dir)
    finally:
        cleanup(zip_path)

if __name__ == "__main__":
    main(Path("data"))
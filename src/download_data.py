"""
Two download sources are tried in order:
  1. UCI's official archive.
  2. a GitHub mirror of the same 8 files.         
"""

import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "har_data"

REQUIRED_FILES = (
    "X_train.txt",
    "X_test.txt",
    "y_train.txt",
    "y_test.txt",
    "subject_train.txt",
    "subject_test.txt",
    "features.txt",
    "activity_labels.txt",
)

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

MIRROR_BASE = (
    "https://raw.githubusercontent.com/anas337/"
    "Human-Activity-Recognition-Using-Smartphones.github.io/master/"
    "Data/Original-Data/UCI-HAR-Dataset/"
)


MIRROR_SUBDIR = {"X_train.txt", "X_test.txt", "features.txt"}


MIN_BYTES = {
    "X_train.txt": 10_000_000,       
    "X_test.txt": 4_000_000,         
    "y_train.txt": 5_000,            
    "y_test.txt": 2_000,            
    "subject_train.txt": 5_000,     
    "subject_test.txt": 2_000,       
    "features.txt": 5_000,           
    "activity_labels.txt": 40,       
}
DEFAULT_MIN_BYTES = 40


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) pca-har-project/1.0"

TIMEOUT = 120


def _fetch(url, dest):
    """Download `url` to `dest`, raising on anything that smells truncated."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        with open(dest, "wb") as handle:
            shutil.copyfileobj(response, handle)

    floor = MIN_BYTES.get(dest.name, DEFAULT_MIN_BYTES)
    size = dest.stat().st_size
    if size < floor:
        raise OSError(f"{dest.name} is only {size} bytes (expected >= {floor})")


def have_all_files(data_dir):
    """True only if all 8 files exist and none is suspiciously small."""
    for name in REQUIRED_FILES:
        path = data_dir / name
        if not path.exists():
            return False
        if path.stat().st_size < MIN_BYTES.get(name, DEFAULT_MIN_BYTES):
            return False
    return True


def _collect_from_tree(tree, data_dir):
    """Copy the 8 required files out of an extracted archive tree.

    Searches recursively by exact filename because the archive nests them
    unevenly: features.txt and activity_labels.txt sit at the top level, while
    y_train.txt lives under train/ and y_test.txt under test/.
    """
    for name in REQUIRED_FILES:
        # sorted() keeps the choice deterministic if an archive somehow carries
        # duplicates (e.g. a __MACOSX shadow copy).
        matches = sorted(tree.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} not found in extracted archive")
        shutil.copy2(matches[0], data_dir / name)


def download_from_uci(data_dir):
    """Primary source: one zip, containing another zip, containing the data."""
    print(f"  trying UCI archive: {UCI_ZIP_URL}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        outer_zip = tmp / "har.zip"
        _fetch(UCI_ZIP_URL, outer_zip)
        print(f"  downloaded {outer_zip.stat().st_size / 1e6:.1f} MB")

        extracted = tmp / "extracted"
        with zipfile.ZipFile(outer_zip) as archive:
            archive.extractall(extracted)

        # Second level: extract every nested zip found, in place.
        nested = [z for z in sorted(extracted.rglob("*.zip"))]
        print(f"  nested zips: {[z.name for z in nested]}")
        for inner_zip in nested:
            with zipfile.ZipFile(inner_zip) as archive:
                archive.extractall(inner_zip.parent / inner_zip.stem)

        _collect_from_tree(extracted, data_dir)


def download_from_mirror(data_dir):
    """Fallback source: the same 8 files, fetched one at a time from GitHub."""
    print(f"  trying mirror: {MIRROR_BASE}")
    for name in REQUIRED_FILES:
        prefix = "Processed-Data/" if name in MIRROR_SUBDIR else ""
        _fetch(MIRROR_BASE + prefix + name, data_dir / name)
        print(f"    fetched {name}")


def report_sizes(data_dir):
    """Print each file's size so a silent truncation is visible, not inferred."""
    print(f"\nfiles in {data_dir}:")
    for name in REQUIRED_FILES:
        size_mb = (data_dir / name).stat().st_size / 1e6
        print(f"  {name:<22} {size_mb:>7.2f} MB")


def ensure_dataset(data_dir=DEFAULT_DATA_DIR):
    """Make sure the 8 dataset files exist locally. Returns the data directory."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if have_all_files(data_dir):
        print(f"dataset already present in {data_dir} — skipping download")
        report_sizes(data_dir)
        return data_dir

    print("downloading UCI HAR dataset...")
    try:
        download_from_uci(data_dir)
    except (urllib.error.URLError, OSError, zipfile.BadZipFile,
            FileNotFoundError) as exc:
        print(f"  primary source failed: {type(exc).__name__}: {exc}")
        print("  falling back to mirror")
        download_from_mirror(data_dir)

    if not have_all_files(data_dir):
        raise RuntimeError(f"download finished but {data_dir} is incomplete")

    report_sizes(data_dir)
    return data_dir


if __name__ == "__main__":
    ensure_dataset()

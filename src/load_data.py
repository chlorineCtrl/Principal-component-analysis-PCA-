"""
Step 2 — parse the flat UCI HAR files into NumPy arrays.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "har_data"


def dedupe_names(names):
    """Suffix repeated feature names with __1, __2, ... in order of appearance.

    features.txt has 42 names that each occur 3 times (the bandsEnergy features,
    one per axis, with the axis missing from the label). The values differ, so
    the columns are kept; only the labels are made unique.
    """
    seen = {}
    unique = []
    for name in names:
        if name in seen:
            unique.append(f"{name}__{seen[name]}")
            seen[name] += 1
        else:
            unique.append(name)
            seen[name] = 1
    return unique


def read_feature_names(path):
    """features.txt holds `index name` pairs; return the 561 deduplicated names."""
    names = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                names.append(line.split(None, 1)[1])
    return dedupe_names(names)


def read_label_map(path):
    """activity_labels.txt holds `id name` pairs; return {1: 'WALKING', ...}."""
    label_map = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                activity_id, name = line.split(None, 1)
                label_map[int(activity_id)] = name
    return label_map


def _read_matrix(path):
    """Whitespace-separated float matrix, no header. pandas for speed over np.loadtxt."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64)
    return frame.to_numpy()


def _read_column(path):
    """Single column of ints."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.int64)
    return frame.to_numpy().ravel()


def load_har(data_dir=DEFAULT_DATA_DIR):
    """Load the dataset. Returns (X_train, X_test, y_train, y_test,
    subj_train, subj_test, feature_names, label_map)."""
    data_dir = Path(data_dir)

    missing = [
        name
        for name in ("X_train.txt", "X_test.txt", "y_train.txt", "y_test.txt",
                     "subject_train.txt", "subject_test.txt", "features.txt",
                     "activity_labels.txt")
        if not (data_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} in {data_dir} — run src/download_data.py first"
        )

    feature_names = read_feature_names(data_dir / "features.txt")
    label_map = read_label_map(data_dir / "activity_labels.txt")

    X_train = _read_matrix(data_dir / "X_train.txt")
    X_test = _read_matrix(data_dir / "X_test.txt")
    y_train = _read_column(data_dir / "y_train.txt")
    y_test = _read_column(data_dir / "y_test.txt")
    subj_train = _read_column(data_dir / "subject_train.txt")
    subj_test = _read_column(data_dir / "subject_test.txt")

    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"X_train has {X_train.shape[1]} columns but features.txt "
            f"lists {len(feature_names)} names"
        )

    return (X_train, X_test, y_train, y_test, subj_train, subj_test,
            feature_names, label_map)


def describe(X_train, X_test, y_train, y_test, subj_train, subj_test,
             feature_names, label_map):
    """Print the shape/split/sanity summary used as this step's acceptance check."""
    print(f"X_train {X_train.shape}   X_test {X_test.shape}")
    print(f"y_train {y_train.shape}      y_test {y_test.shape}")
    print(f"subj_train {subj_train.shape}   subj_test {subj_test.shape}")
    print(f"feature names: {len(feature_names)} "
          f"({len(set(feature_names))} unique after dedupe)")

    train_subjects = sorted(set(subj_train.tolist()))
    test_subjects = sorted(set(subj_test.tolist()))
    print(f"\ntrain subjects ({len(train_subjects)}): {train_subjects}")
    print(f"test  subjects ({len(test_subjects)}):  {test_subjects}")
    print("overlap between train and test subjects: "
          f"{set(train_subjects) & set(test_subjects)}   <- must be empty")

    print(f"\nfeature value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"missing values: train={np.isnan(X_train).sum()}, "
          f"test={np.isnan(X_test).sum()}")

    print(f"\nlabel map: {label_map}")
    print(f"classes present in y_train: {sorted(set(y_train.tolist()))}")


if __name__ == "__main__":
    describe(*load_har())

"""Resolve data paths and check input alignment without changing preprocessing."""
import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = ("train_features.csv", "train_targets_scored.csv")


def data_directory(directory=None):
    value = directory if directory is not None else os.environ.get("MOA_DATA_DIR")
    return Path(value).expanduser().resolve() if value else PROJECT_ROOT / "data" / "raw"


def validate_data(directory=None):
    """Check both CSVs, binary labels, and matching unique IDs in original row order."""
    folder = data_directory(directory)
    ids = []
    counts = []
    for filename in REQUIRED_FILES:
        path = folder / filename
        if not path.is_file():
            raise ValueError(f"Missing {path}. Extract the Kaggle lish-moa data into {folder}, "
                             "or set MOA_DATA_DIR to your CSV directory.")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            required = {"sig_id", "cp_type", "cp_time", "cp_dose"} if filename == REQUIRED_FILES[0] else {"sig_id"}
            if not required.issubset(header) or len(header) != len(set(header)):
                raise ValueError(f"{filename}: missing required columns or duplicate column names.")
            if len(header) <= len(required):
                raise ValueError(f"{filename}: no feature/target columns found.")
            id_index = header.index("sig_id")
            file_ids = []
            for line, row in enumerate(reader, 2):
                if len(row) != len(header) or not row[id_index]:
                    raise ValueError(f"{filename}: invalid row at line {line}.")
                if filename == REQUIRED_FILES[1] and any(
                    v not in {"0", "1", "0.0", "1.0"} for i, v in enumerate(row) if i != id_index
                ):
                    raise ValueError(f"{filename}: non-binary target at line {line}.")
                file_ids.append(row[id_index])
            if not file_ids or len(file_ids) != len(set(file_ids)):
                raise ValueError(f"{filename}: empty data or duplicate sig_id values.")
            ids.append(file_ids)
            counts.append(len(header) - 1)
    if ids[0] != ids[1]:
        raise ValueError("Feature and target sig_id values must match in the same row order.")
    return {"directory": str(folder), "rows": len(ids[0]), "features": counts[0], "targets": counts[1]}


def load_training_data(directory=None):
    validate_data(directory)
    import pandas as pd
    folder = data_directory(directory)
    return tuple(pd.read_csv(folder / name) for name in REQUIRED_FILES)

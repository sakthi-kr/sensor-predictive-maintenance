

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "cwru"

ALLOWED_LABELS = {
    "normal",
    "inner_race_fault",
    "ball_fault",
    "outer_race_fault",
}

@dataclass
class BearingSignal:
    file_path: Path
    label: str
    signal: np.ndarray
    signal_key: str
    rpm: Optional[float] = None



def find_mat_files(data_root: Path = DEFAULT_DATA_ROOT) -> list[Path]:
    """
    Find all .mat files inside the dataset folder.

    Expected structure:

    data/raw/cwru/
        normal/
        inner_race_fault/
        ball_fault/
        outer_race_fault/
    """
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    mat_files = sorted(data_root.rglob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found inside: {data_root}")

    return mat_files






def label_from_path(file_path: Path, data_root: Path = DEFAULT_DATA_ROOT) -> str:
    """
    Infer the class label from the folder name.

    Example:
    data/raw/cwru/normal/97.mat -> normal
    data/raw/cwru/ball_fault/118.mat -> ball_fault
    """
    file_path = Path(file_path)
    data_root = Path(data_root)

    relative_parts = file_path.relative_to(data_root).parts
    label = relative_parts[0]

    if label not in ALLOWED_LABELS:
        raise ValueError(
            f"Unknown label '{label}' from file path: {file_path}\n"
            f"Allowed labels are: {sorted(ALLOWED_LABELS)}"
        )

    return label


def select_vibration_key(mat_data: dict, preferred_sensor: str = "DE") -> str:
    """
    Select the vibration signal key from a loaded CWRU .mat file.

    CWRU keys usually look like:
    X097_DE_time
    X097_FE_time
    X097RPM

    preferred_sensor:
    - "DE" = drive-end accelerometer
    - "FE" = fan-end accelerometer

    For this project, we start with DE because drive-end bearing data is commonly used.
    """
    preferred_sensor = preferred_sensor.upper()

    if preferred_sensor not in {"DE", "FE"}:
        raise ValueError("preferred_sensor must be either 'DE' or 'FE'")

    preferred_suffix = f"{preferred_sensor}_time"

    # First choice: preferred sensor, usually DE_time
    for key in mat_data.keys():
        if key.endswith(preferred_suffix):
            return key

    # Fallback: any vibration time signal
    for fallback_suffix in ["DE_time", "FE_time"]:
        for key in mat_data.keys():
            if key.endswith(fallback_suffix):
                return key

    available_keys = [key for key in mat_data.keys() if not key.startswith("__")]
    raise KeyError(
        "Could not find a vibration signal key ending in DE_time or FE_time.\n"
        f"Available keys: {available_keys}"
    )


def extract_rpm(mat_data: dict) -> Optional[float]:
    """
    Extract RPM value if it exists.

    Some CWRU files contain a key like X097RPM.
    """
    for key, value in mat_data.items():
        if key.endswith("RPM"):
            rpm_array = np.asarray(value).squeeze()
            if rpm_array.size > 0:
                return float(rpm_array)

    return None


def load_bearing_signal(
    file_path: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
    preferred_sensor: str = "DE",
) -> BearingSignal:
    """
    Load one .mat file and return a BearingSignal object.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    mat_data = loadmat(file_path)

    label = label_from_path(file_path, data_root=data_root)
    signal_key = select_vibration_key(mat_data, preferred_sensor=preferred_sensor)
    rpm = extract_rpm(mat_data)

    signal = np.asarray(mat_data[signal_key]).squeeze().astype(float)

    if signal.ndim != 1:
        raise ValueError(
            f"Expected a 1D vibration signal, but got shape {signal.shape} "
            f"from key {signal_key} in file {file_path}"
        )

    if signal.size == 0:
        raise ValueError(f"Empty signal found in file: {file_path}")

    return BearingSignal(
        file_path=file_path,
        label=label,
        signal=signal,
        signal_key=signal_key,
        rpm=rpm,
    )


def load_dataset(
    data_root: Path = DEFAULT_DATA_ROOT,
    preferred_sensor: str = "DE",
) -> list[BearingSignal]:
    """
    Load all .mat files from the local CWRU dataset folder.
    """
    mat_files = find_mat_files(data_root)

    records = [
        load_bearing_signal(
            file_path=file_path,
            data_root=data_root,
            preferred_sensor=preferred_sensor,
        )
        for file_path in mat_files
    ]

    return records


def print_dataset_summary(records: list[BearingSignal]) -> None:
    """
    Print a simple summary of loaded files.
    """
    print(f"\nLoaded {len(records)} files.\n")

    for record in records:
        print(f"File: {record.file_path.name}")
        print(f"  Label      : {record.label}")
        print(f"  Signal key : {record.signal_key}")
        print(f"  Signal size: {record.signal.size}")
        print(f"  RPM        : {record.rpm}")
        print(f"  Mean       : {record.signal.mean():.6f}")
        print(f"  Std        : {record.signal.std():.6f}")
        print()


if __name__ == "__main__":
    dataset = load_dataset()
    print_dataset_summary(dataset)
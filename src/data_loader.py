from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

try:
    from src.dataset_manifest import (
        CWRUFileMetadata,
        DEFAULT_MANIFEST_PATH,
        load_manifest,
    )
except ModuleNotFoundError:
    from dataset_manifest import (
        CWRUFileMetadata,
        DEFAULT_MANIFEST_PATH,
        load_manifest,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cwru"
)

TARGET_SAMPLE_RATE_HZ = 12000

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
    source_recording: Optional[str] = None
    fault_type: Optional[str] = None
    fault_diameter_in: Optional[float] = None
    outer_race_position: Optional[str] = None
    load_hp: Optional[int] = None
    approx_rpm: Optional[int] = None
    original_sample_rate_hz: int = (
        TARGET_SAMPLE_RATE_HZ
    )
    sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ
    dataset_section: Optional[str] = None


def find_mat_files(
    data_root: Path = DEFAULT_DATA_ROOT,
) -> list[Path]:
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root does not exist: {data_root}"
        )

    mat_files = sorted(
        data_root.rglob("*.mat")
    )

    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found inside: {data_root}"
        )

    return mat_files


def label_from_path(
    file_path: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> str:
    file_path = Path(file_path)
    data_root = Path(data_root)

    relative_parts = (
        file_path
        .relative_to(data_root)
        .parts
    )

    label = relative_parts[0]

    if label not in ALLOWED_LABELS:
        raise ValueError(
            f"Unknown label '{label}' from file path: "
            f"{file_path}\n"
            f"Allowed labels are: "
            f"{sorted(ALLOWED_LABELS)}"
        )

    return label


def select_vibration_key(
    mat_data: dict,
    preferred_sensor: str = "DE",
) -> str:
    preferred_sensor = (
        preferred_sensor.upper()
    )

    if preferred_sensor not in {
        "DE",
        "FE",
    }:
        raise ValueError(
            "preferred_sensor must be either "
            "'DE' or 'FE'"
        )

    preferred_suffix = (
        f"{preferred_sensor}_time"
    )

    for key in mat_data:
        if key.endswith(preferred_suffix):
            return key

    for fallback_suffix in [
        "DE_time",
        "FE_time",
    ]:
        for key in mat_data:
            if key.endswith(
                fallback_suffix
            ):
                return key

    available_keys = [
        key
        for key in mat_data
        if not key.startswith("__")
    ]

    raise KeyError(
        "Could not find a vibration signal "
        "key ending in DE_time or FE_time.\n"
        f"Available keys: {available_keys}"
    )


def extract_rpm(
    mat_data: dict,
) -> Optional[float]:
    for key, value in mat_data.items():
        if key.endswith("RPM"):
            rpm_array = (
                np.asarray(value)
                .squeeze()
            )

            if rpm_array.size > 0:
                return float(rpm_array)

    return None


def standardize_sample_rate(
    signal: np.ndarray,
    original_sample_rate_hz: int,
    target_sample_rate_hz: int,
) -> np.ndarray:
    signal = (
        np.asarray(
            signal,
            dtype=float,
        )
        .squeeze()
    )

    if signal.ndim != 1:
        raise ValueError(
            "Expected a 1D signal, "
            f"got shape {signal.shape}"
        )

    if original_sample_rate_hz <= 0:
        raise ValueError(
            "original_sample_rate_hz "
            "must be positive"
        )

    if target_sample_rate_hz <= 0:
        raise ValueError(
            "target_sample_rate_hz "
            "must be positive"
        )

    if (
        original_sample_rate_hz
        == target_sample_rate_hz
    ):
        return signal.copy()

    divisor = math.gcd(
        original_sample_rate_hz,
        target_sample_rate_hz,
    )

    up = (
        target_sample_rate_hz
        // divisor
    )

    down = (
        original_sample_rate_hz
        // divisor
    )

    standardized = resample_poly(
        signal,
        up=up,
        down=down,
    )

    return np.asarray(
        standardized,
        dtype=float,
    )


def load_bearing_signal(
    file_path: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
    preferred_sensor: str = "DE",
    metadata: Optional[
        CWRUFileMetadata
    ] = None,
    target_sample_rate_hz: Optional[
        int
    ] = None,
) -> BearingSignal:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    mat_data = loadmat(file_path)

    label = label_from_path(
        file_path,
        data_root=data_root,
    )

    if (
        metadata is not None
        and metadata.label != label
    ):
        raise ValueError(
            "Manifest label does not match "
            f"folder label for {file_path}: "
            f"manifest={metadata.label}, "
            f"folder={label}"
        )

    signal_key = select_vibration_key(
        mat_data,
        preferred_sensor=preferred_sensor,
    )

    rpm = extract_rpm(mat_data)

    raw_signal = (
        np.asarray(
            mat_data[signal_key]
        )
        .squeeze()
        .astype(float)
    )

    if raw_signal.ndim != 1:
        raise ValueError(
            "Expected a 1D vibration signal, "
            f"but got shape {raw_signal.shape} "
            f"from key {signal_key} "
            f"in file {file_path}"
        )

    if raw_signal.size == 0:
        raise ValueError(
            f"Empty signal found in file: "
            f"{file_path}"
        )

    original_sample_rate_hz = (
        metadata.sample_rate_hz
        if metadata is not None
        else TARGET_SAMPLE_RATE_HZ
    )

    effective_sample_rate_hz = (
        original_sample_rate_hz
        if target_sample_rate_hz is None
        else target_sample_rate_hz
    )

    signal = standardize_sample_rate(
        raw_signal,
        original_sample_rate_hz=(
            original_sample_rate_hz
        ),
        target_sample_rate_hz=(
            effective_sample_rate_hz
        ),
    )

    return BearingSignal(
        file_path=file_path,
        label=label,
        signal=signal,
        signal_key=signal_key,
        rpm=rpm,
        source_recording=(
            metadata.file_name
            if metadata is not None
            else file_path.name
        ),
        fault_type=(
            metadata.fault_type
            if metadata is not None
            else label
        ),
        fault_diameter_in=(
            metadata.fault_diameter_in
            if metadata is not None
            else None
        ),
        outer_race_position=(
            metadata.outer_race_position
            if metadata is not None
            else None
        ),
        load_hp=(
            metadata.load_hp
            if metadata is not None
            else None
        ),
        approx_rpm=(
            metadata.approx_rpm
            if metadata is not None
            else None
        ),
        original_sample_rate_hz=(
            original_sample_rate_hz
        ),
        sample_rate_hz=(
            effective_sample_rate_hz
        ),
        dataset_section=(
            metadata.dataset_section
            if metadata is not None
            else None
        ),
    )


def load_dataset(
    data_root: Path = DEFAULT_DATA_ROOT,
    preferred_sensor: str = "DE",
) -> list[BearingSignal]:
    mat_files = find_mat_files(data_root)

    return [
        load_bearing_signal(
            file_path=file_path,
            data_root=data_root,
            preferred_sensor=(
                preferred_sensor
            ),
        )
        for file_path in mat_files
    ]


def load_dataset_from_manifest(
    manifest_path: Path = (
        DEFAULT_MANIFEST_PATH
    ),
    preferred_sensor: str = "DE",
    target_sample_rate_hz: int = (
        TARGET_SAMPLE_RATE_HZ
    ),
) -> list[BearingSignal]:
    metadata_records = load_manifest(
        manifest_path
    )

    records: list[BearingSignal] = []

    for metadata in metadata_records:
        if not metadata.local_path.exists():
            raise FileNotFoundError(
                "Dataset file listed in the "
                "manifest is missing: "
                f"{metadata.local_path}"
            )

        records.append(
            load_bearing_signal(
                file_path=(
                    metadata.local_path
                ),
                data_root=DEFAULT_DATA_ROOT,
                preferred_sensor=(
                    preferred_sensor
                ),
                metadata=metadata,
                target_sample_rate_hz=(
                    target_sample_rate_hz
                ),
            )
        )

    return records


def print_dataset_summary(
    records: list[BearingSignal],
) -> None:
    print(
        f"\nLoaded {len(records)} files.\n"
    )

    for record in records:
        print(
            f"File: {record.file_path.name}"
        )
        print(
            f"  Label        : "
            f"{record.label}"
        )
        print(
            f"  Signal key   : "
            f"{record.signal_key}"
        )
        print(
            f"  Signal size  : "
            f"{record.signal.size}"
        )
        print(
            f"  Recorded RPM : "
            f"{record.rpm}"
        )
        print(
            f"  Manifest load: "
            f"{record.load_hp} HP"
        )
        print(
            "  Sample rate  : "
            f"{record.original_sample_rate_hz} "
            "-> "
            f"{record.sample_rate_hz} Hz"
        )
        print(
            f"  Mean         : "
            f"{record.signal.mean():.6f}"
        )
        print(
            f"  Std          : "
            f"{record.signal.std():.6f}"
        )
        print()


if __name__ == "__main__":
    dataset = load_dataset_from_manifest()
    print_dataset_summary(dataset)

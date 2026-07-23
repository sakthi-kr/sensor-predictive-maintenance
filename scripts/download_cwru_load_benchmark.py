from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "cwru_load_benchmark.csv"
)

DOWNLOAD_TIMEOUT_SECONDS = 180
DOWNLOAD_CHUNK_SIZE_BYTES = 1024 * 1024

REQUIRED_MANIFEST_COLUMNS = {
    "file_name",
    "label",
    "fault_type",
    "fault_diameter_in",
    "outer_race_position",
    "load_hp",
    "approx_rpm",
    "sample_rate_hz",
    "dataset_section",
    "local_path",
    "download_url",
}


@dataclass(frozen=True)
class DatasetFile:
    file_name: str
    label: str
    fault_type: str
    fault_diameter_in: float
    outer_race_position: str
    load_hp: int
    approx_rpm: int
    sample_rate_hz: int
    dataset_section: str
    local_path: Path
    download_url: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and validate the 16-file CWRU "
            "load-generalization benchmark."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the CSV dataset manifest.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download files again even when valid local files exist.",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate existing files without downloading anything.",
    )

    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def read_manifest(manifest_path: Path) -> list[DatasetFile]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"Manifest has no header: {manifest_path}"
            )

        missing_columns = (
            REQUIRED_MANIFEST_COLUMNS
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        records = []

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                record = DatasetFile(
                    file_name=row["file_name"].strip(),
                    label=row["label"].strip(),
                    fault_type=row["fault_type"].strip(),
                    fault_diameter_in=float(
                        row["fault_diameter_in"]
                    ),
                    outer_race_position=(
                        row["outer_race_position"].strip()
                    ),
                    load_hp=int(row["load_hp"]),
                    approx_rpm=int(row["approx_rpm"]),
                    sample_rate_hz=int(
                        row["sample_rate_hz"]
                    ),
                    dataset_section=(
                        row["dataset_section"].strip()
                    ),
                    local_path=resolve_project_path(
                        row["local_path"].strip()
                    ),
                    download_url=(
                        row["download_url"].strip()
                    ),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Invalid value in manifest row "
                    f"{row_number}: {error}"
                ) from error

            records.append(record)

    if len(records) != 16:
        raise ValueError(
            "The load benchmark manifest should contain "
            f"exactly 16 files, but contains {len(records)}."
        )

    file_names = [
        record.file_name
        for record in records
    ]

    if len(file_names) != len(set(file_names)):
        raise ValueError(
            "Duplicate file names were found in the manifest."
        )

    return records


def find_vibration_keys(
    mat_data: dict,
) -> list[str]:
    return sorted(
        key
        for key in mat_data
        if key.endswith("_time")
        and not key.startswith("__")
    )


def find_drive_end_keys(
    mat_data: dict,
) -> list[str]:
    return sorted(
        key
        for key in mat_data
        if key.endswith("DE_time")
        and not key.startswith("__")
    )


def validate_mat_file(
    file_path: Path,
) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Downloaded file does not exist: {file_path}"
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"Downloaded file is empty: {file_path}"
        )

    try:
        mat_data = loadmat(file_path)
    except Exception as error:
        raise ValueError(
            f"File is not a readable MATLAB file: {file_path}"
        ) from error

    vibration_keys = find_vibration_keys(mat_data)
    drive_end_keys = find_drive_end_keys(mat_data)

    if not vibration_keys:
        available_keys = sorted(
            key
            for key in mat_data
            if not key.startswith("__")
        )

        raise ValueError(
            "No vibration time-series key was found in "
            f"{file_path}. Available keys: {available_keys}"
        )

    if not drive_end_keys:
        raise ValueError(
            "No drive-end vibration key ending in "
            f"'DE_time' was found in {file_path}. "
            f"Vibration keys: {vibration_keys}"
        )

    signal = mat_data[drive_end_keys[0]].squeeze()

    if signal.ndim != 1:
        raise ValueError(
            "Expected one-dimensional drive-end signal in "
            f"{file_path}, but received shape {signal.shape}."
        )

    if signal.size == 0:
        raise ValueError(
            f"Drive-end signal is empty in {file_path}."
        )

    rpm_keys = sorted(
        key
        for key in mat_data
        if key.endswith("RPM")
    )

    return {
        "size_bytes": file_path.stat().st_size,
        "drive_end_key": drive_end_keys[0],
        "signal_samples": int(signal.size),
        "rpm_key": rpm_keys[0] if rpm_keys else None,
    }


def download_file(
    url: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 CWRU-bearing-data-downloader"
            )
        },
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f"{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

            with urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                content_type = (
                    response.headers
                    .get("Content-Type", "")
                    .lower()
                )

                if "text/html" in content_type:
                    raise ValueError(
                        "The server returned an HTML page "
                        f"instead of a MATLAB file for {url}"
                    )

                shutil.copyfileobj(
                    response,
                    temporary_file,
                    length=DOWNLOAD_CHUNK_SIZE_BYTES,
                )

            temporary_file.flush()

        # The temporary file is now closed. This is required
        # on Windows before loading, renaming, or deleting it.
        validate_mat_file(temporary_path)

        if destination.exists():
            destination.unlink()

        temporary_path.replace(destination)

    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(
                    missing_ok=True,
                )
            except PermissionError:
                print(
                    "WARNING: Could not immediately delete "
                    f"temporary file: {temporary_path}",
                    file=sys.stderr,
                )

        raise


def format_megabytes(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def process_record(
    record: DatasetFile,
    *,
    overwrite: bool,
    verify_only: bool,
) -> dict:
    path = record.local_path

    should_download = (
        overwrite
        or not path.exists()
    )

    if verify_only and not path.exists():
        raise FileNotFoundError(
            f"Required dataset file is missing: {path}"
        )

    if should_download:
        if verify_only:
            raise FileNotFoundError(
                f"Cannot verify missing file: {path}"
            )

        print(
            f"Downloading {record.file_name} "
            f"({record.label}, load={record.load_hp} HP)"
        )

        download_file(
            record.download_url,
            path,
        )

    else:
        print(
            f"Using existing {record.file_name} "
            f"({record.label}, load={record.load_hp} HP)"
        )

    validation = validate_mat_file(path)

    return {
        "file_name": record.file_name,
        "label": record.label,
        "load_hp": record.load_hp,
        "approx_rpm": record.approx_rpm,
        "fault_diameter_in": (
            record.fault_diameter_in
        ),
        "local_path": path,
        **validation,
    }


def print_summary(
    results: list[dict],
) -> None:
    print()
    print("=" * 78)
    print("CWRU LOAD BENCHMARK VERIFICATION")
    print("=" * 78)

    for result in results:
        print(
            f"{result['file_name']:>7} | "
            f"{result['label']:<18} | "
            f"load={result['load_hp']} HP | "
            f"samples={result['signal_samples']:>7} | "
            f"{format_megabytes(result['size_bytes']):>9} | "
            f"key={result['drive_end_key']}"
        )

    print()
    print(f"Verified files: {len(results)}")

    total_bytes = sum(
        result["size_bytes"]
        for result in results
    )

    print(
        "Total local size: "
        f"{format_megabytes(total_bytes)}"
    )

    class_counts: dict[str, int] = {}
    load_counts: dict[int, int] = {}

    for result in results:
        class_counts[result["label"]] = (
            class_counts.get(
                result["label"],
                0,
            )
            + 1
        )

        load_counts[result["load_hp"]] = (
            load_counts.get(
                result["load_hp"],
                0,
            )
            + 1
        )

    print()
    print("Files per class:")

    for label in sorted(class_counts):
        print(
            f"  {label:<18}: "
            f"{class_counts[label]}"
        )

    print()
    print("Files per load:")

    for load_hp in sorted(load_counts):
        print(
            f"  {load_hp} HP: "
            f"{load_counts[load_hp]}"
        )

    expected_class_count = 4
    expected_load_count = 4

    if any(
        count != expected_class_count
        for count in class_counts.values()
    ):
        raise ValueError(
            "The benchmark is not balanced across classes."
        )

    if any(
        count != expected_load_count
        for count in load_counts.values()
    ):
        raise ValueError(
            "The benchmark is not balanced across loads."
        )

    print()
    print(
        "PASS: All 16 files are readable and the "
        "benchmark is balanced across classes and loads."
    )


def main() -> int:
    arguments = parse_arguments()

    try:
        records = read_manifest(
            arguments.manifest,
        )

        results = [
            process_record(
                record,
                overwrite=arguments.overwrite,
                verify_only=arguments.verify_only,
            )
            for record in records
        ]

        print_summary(results)

    except (
        FileNotFoundError,
        ValueError,
        urllib.error.URLError,
        TimeoutError,
    ) as error:
        print()
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

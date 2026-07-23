from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "cwru_load_benchmark.csv"
)

REQUIRED_COLUMNS = {
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
class CWRUFileMetadata:
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

    def to_dict(self) -> dict:
        data = asdict(self)
        data["local_path"] = str(self.local_path)
        return data


def resolve_project_path(
    path_value: str | Path,
) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> list[CWRUFileMetadata]:
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Dataset manifest does not exist: "
            f"{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Dataset manifest has no header: "
                f"{manifest_path}"
            )

        missing_columns = (
            REQUIRED_COLUMNS
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Dataset manifest is missing required "
                f"columns: {sorted(missing_columns)}"
            )

        records: list[CWRUFileMetadata] = []

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                file_name = row["file_name"].strip()

                local_path = resolve_project_path(
                    row["local_path"].strip()
                )

                if local_path.name != file_name:
                    raise ValueError(
                        "file_name and local_path filename "
                        "do not match"
                    )

                sample_rate_hz = int(
                    row["sample_rate_hz"]
                )

                if sample_rate_hz <= 0:
                    raise ValueError(
                        "sample_rate_hz must be positive"
                    )

                record = CWRUFileMetadata(
                    file_name=file_name,
                    label=row["label"].strip(),
                    fault_type=(
                        row["fault_type"].strip()
                    ),
                    fault_diameter_in=float(
                        row["fault_diameter_in"]
                    ),
                    outer_race_position=(
                        row[
                            "outer_race_position"
                        ].strip()
                    ),
                    load_hp=int(row["load_hp"]),
                    approx_rpm=int(
                        row["approx_rpm"]
                    ),
                    sample_rate_hz=sample_rate_hz,
                    dataset_section=(
                        row[
                            "dataset_section"
                        ].strip()
                    ),
                    local_path=local_path,
                    download_url=(
                        row["download_url"].strip()
                    ),
                )

            except (
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    "Invalid manifest row "
                    f"{row_number}: {error}"
                ) from error

            records.append(record)

    if not records:
        raise ValueError(
            "Dataset manifest contains no records: "
            f"{manifest_path}"
        )

    file_names = [
        record.file_name
        for record in records
    ]

    if len(file_names) != len(set(file_names)):
        raise ValueError(
            "Dataset manifest contains duplicate "
            "file names."
        )

    local_paths = [
        record.local_path
        for record in records
    ]

    if len(local_paths) != len(set(local_paths)):
        raise ValueError(
            "Dataset manifest contains duplicate "
            "local paths."
        )

    return records


def manifest_by_file_name(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, CWRUFileMetadata]:
    return {
        record.file_name: record
        for record in load_manifest(
            manifest_path
        )
    }

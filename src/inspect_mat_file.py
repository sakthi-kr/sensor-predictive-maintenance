from pathlib import Path
from scipy.io import loadmat


def inspect_mat_file(file_path: str) -> None:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = loadmat(path)

    print(f"\nFile: {path}")
    print("Available keys:")

    for key, value in data.items():
        if not key.startswith("__"):
            shape = getattr(value, "shape", None)
            print(f"  {key}: shape={shape}")


if __name__ == "__main__":
    inspect_mat_file("data/raw/cwru/normal/97.mat")

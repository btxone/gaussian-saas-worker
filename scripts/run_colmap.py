import subprocess
from pathlib import Path


def run_colmap(dataset_dir: Path) -> Path:
    gaussian_root = Path("/opt/gaussian-splatting")
    convert_script = gaussian_root / "convert.py"
    subprocess.run(
        [
            "python3",
            str(convert_script),
            "-s",
            str(dataset_dir),
            "--resize",
        ],
        check=True,
    )
    sparse_dir = dataset_dir / "sparse"
    sparse_model_dir = sparse_dir / "0"
    if not sparse_model_dir.exists():
        raise RuntimeError(f"COLMAP did not generate a sparse model at {sparse_model_dir}.")
    return sparse_dir

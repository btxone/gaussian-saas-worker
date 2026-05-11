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
    return dataset_dir / "sparse"

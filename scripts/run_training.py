import subprocess
from pathlib import Path


def run_training(dataset_dir: Path, model_dir: Path, settings: dict) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    iterations = int(settings.get("iterations", 7000))
    train_script = Path("/opt/gaussian-splatting/train.py")
    subprocess.run(
        [
            "python3",
            str(train_script),
            "-s",
            str(dataset_dir),
            "-m",
            str(model_dir),
            "--iterations",
            str(iterations),
            "--quiet",
        ],
        check=True,
    )

    point_cloud_root = model_dir / "point_cloud"
    candidates = sorted(point_cloud_root.glob("iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError("Gaussian Splatting training finished without point_cloud.ply output.")
    return candidates[-1]

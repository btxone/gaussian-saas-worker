import subprocess
from pathlib import Path


def run_training(dataset_dir: Path, model_dir: Path, settings: dict) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    iterations = max(int(settings.get("iterations", 30000)), 30000)
    train_script = Path("/opt/gaussian-splatting/train.py")
    _print_torch_cuda_status()
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


def _print_torch_cuda_status() -> None:
    result = subprocess.run(
        [
            "python3",
            "-c",
            "import json, torch; print(json.dumps({'message':'torch_cuda','available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip() or result.stderr.strip()
    if output:
        print(output, flush=True)

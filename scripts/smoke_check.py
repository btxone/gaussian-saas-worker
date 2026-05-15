import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


GAUSSIAN_ROOT = Path(os.getenv("GAUSSIAN_SPLATTING_ROOT", "/opt/gaussian-splatting"))
APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))


def main() -> None:
    _check_commands(["python3", "ffmpeg", "ffprobe", "colmap", "splat-transform"])
    _check_imports(
        [
            "boto3",
            "cv2",
            "diff_gaussian_rasterization",
            "fused_ssim",
            "httpx",
            "joblib",
            "numpy",
            "PIL.Image",
            "plyfile",
            "runpod",
            "simple_knn._C",
            "torch",
            "torchvision",
            "tqdm",
            "scripts.convert_outputs",
            "scripts.extract_frames",
            "scripts.gpu_telemetry",
            "scripts.normalize_video",
            "scripts.run_colmap",
            "scripts.run_training",
            "scripts.storage",
        ]
    )
    _check_graphdeco_imports()
    _check_command_runs()
    print("worker_smoke_check passed", flush=True)


def _check_commands(commands: list[str]) -> None:
    missing = [command for command in commands if shutil.which(command) is None]
    if missing:
        raise RuntimeError(f"Missing required commands: {', '.join(missing)}")


def _check_imports(modules: list[str]) -> None:
    for module in modules:
        importlib.import_module(module)


def _check_graphdeco_imports() -> None:
    if not (GAUSSIAN_ROOT / "train.py").exists():
        raise RuntimeError(f"Graphdeco train.py not found at {GAUSSIAN_ROOT}")
    sys.path.insert(0, str(GAUSSIAN_ROOT))
    for module in [
        "arguments",
        "gaussian_renderer",
        "scene",
        "scene.cameras",
        "utils.camera_utils",
        "utils.graphics_utils",
        "utils.image_utils",
        "utils.loss_utils",
        "train",
    ]:
        importlib.import_module(module)


def _check_command_runs() -> None:
    for command in [
        ["ffmpeg", "-version"],
        ["ffprobe", "-version"],
        ["colmap", "-h"],
    ]:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)


if __name__ == "__main__":
    main()

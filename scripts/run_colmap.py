import shutil
import subprocess
import os
from pathlib import Path


class ColmapError(RuntimeError):
    pass


def run_colmap(dataset_dir: Path) -> Path:
    images_dir = dataset_dir / "input"
    if not images_dir.exists():
        raise ColmapError(f"COLMAP input images directory does not exist: {images_dir}")

    image_count = len(list(images_dir.glob("*.jpg")))
    if image_count < 12:
        raise ColmapError(f"COLMAP needs more frames. Found {image_count} jpg frames in {images_dir}.")

    distorted_dir = dataset_dir / "distorted"
    database_path = distorted_dir / "database.db"
    distorted_sparse_dir = distorted_dir / "sparse"
    sparse_model_dir = distorted_sparse_dir / "0"
    distorted_sparse_dir.mkdir(parents=True, exist_ok=True)

    logs: list[str] = []
    _run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "SIMPLE_RADIAL",
            "--SiftExtraction.max_image_size",
            "1600",
            "--SiftExtraction.max_num_features",
            "8192",
            "--SiftExtraction.peak_threshold",
            "0.003",
            "--SiftExtraction.estimate_affine_shape",
            "1",
            "--SiftExtraction.domain_size_pooling",
            "1",
            "--SiftExtraction.use_gpu",
            "0",
        ],
        logs,
    )

    _run(
        [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(database_path),
            "--SequentialMatching.overlap",
            "20",
            "--SiftMatching.guided_matching",
            "1",
            "--SiftMatching.use_gpu",
            "0",
        ],
        logs,
    )
    _run_mapper(database_path, images_dir, distorted_sparse_dir, logs)

    if not _has_sparse_model(sparse_model_dir):
        _reset_sparse_dir(distorted_sparse_dir)
        _run(
            [
                "colmap",
                "exhaustive_matcher",
                "--database_path",
                str(database_path),
                "--SiftMatching.guided_matching",
                "1",
                "--SiftMatching.use_gpu",
                "0",
            ],
            logs,
        )
        _run_mapper(database_path, images_dir, distorted_sparse_dir, logs)

    if not _has_sparse_model(sparse_model_dir):
        raise ColmapError(_reconstruction_failure_message(sparse_model_dir, image_count, logs))

    _run(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            str(images_dir),
            "--input_path",
            str(sparse_model_dir),
            "--output_path",
            str(dataset_dir),
            "--output_type",
            "COLMAP",
        ],
        logs,
    )
    _normalize_sparse_layout(dataset_dir)
    return dataset_dir / "sparse"


def _run_mapper(database_path: Path, images_dir: Path, sparse_dir: Path, logs: list[str]) -> None:
    _run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_dir),
            "--Mapper.min_num_matches",
            "4",
            "--Mapper.init_min_num_inliers",
            "15",
            "--Mapper.init_max_error",
            "8",
            "--Mapper.abs_pose_min_num_inliers",
            "8",
            "--Mapper.abs_pose_max_error",
            "12",
            "--Mapper.filter_max_reproj_error",
            "8",
            "--Mapper.ba_global_function_tolerance",
            "0.000001",
        ],
        logs,
        check=False,
    )


def _run(command: list[str], logs: list[str], check: bool = True) -> int:
    print(f"Running command: {' '.join(command)}", flush=True)
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
    Path(env["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    if output:
        logs.append(f"$ {' '.join(command)}\n{output}")
        print(_tail([output], max_chars=3000), flush=True)
    if check and result.returncode != 0:
        raise ColmapError(f"COLMAP command failed with exit code {result.returncode}: {' '.join(command)}\n{_tail(logs)}")
    return result.returncode


def _normalize_sparse_layout(dataset_dir: Path) -> None:
    sparse_dir = dataset_dir / "sparse"
    sparse_zero_dir = sparse_dir / "0"
    sparse_zero_dir.mkdir(parents=True, exist_ok=True)
    for name in ["cameras.bin", "images.bin", "points3D.bin"]:
        source = sparse_dir / name
        if source.exists():
            shutil.move(str(source), str(sparse_zero_dir / name))


def _reset_sparse_dir(sparse_dir: Path) -> None:
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)


def _has_sparse_model(sparse_model_dir: Path) -> bool:
    return all((sparse_model_dir / name).exists() for name in ["cameras.bin", "images.bin", "points3D.bin"])


def _reconstruction_failure_message(sparse_model_dir: Path, image_count: int, logs: list[str]) -> str:
    combined_logs = "\n".join(logs)
    if "No good initial image pair found" in combined_logs:
        return (
            "COLMAP could not initialize a sparse reconstruction after sequential and exhaustive matching. "
            f"It processed {image_count} frames but could not find a reliable initial image pair. "
            "This often happens with short videos, pure rotation, motion blur, repeated walls, or too little sideways/parallax movement. "
            "Try a slower 60-90 second walkthrough with lateral movement around objects, visible corners/furniture, and strong overlap between views. "
            f"Log tail:\n{_tail(logs, max_chars=900)}"
        )
    return (
        f"COLMAP did not generate a sparse model at {sparse_model_dir} after processing {image_count} frames. "
        f"Log tail:\n{_tail(logs, max_chars=900)}"
    )


def _tail(logs: list[str], max_chars: int = 8000) -> str:
    return "\n".join(logs)[-max_chars:]

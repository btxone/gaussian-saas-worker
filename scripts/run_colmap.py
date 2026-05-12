import shutil
import subprocess
import os
import re
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
            "OPENCV",
            "--SiftExtraction.max_image_size",
            "1920",
            "--SiftExtraction.max_num_features",
            "16384",
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
            "50",
            "--SiftMatching.guided_matching",
            "1",
            "--SiftMatching.use_gpu",
            "0",
        ],
        logs,
    )
    sparse_model_dir = _run_best_mapper(database_path, images_dir, distorted_sparse_dir, logs, image_count)

    if sparse_model_dir is None:
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
        sparse_model_dir = _run_best_mapper(database_path, images_dir, distorted_sparse_dir, logs, image_count)

    if sparse_model_dir is None:
        raise ColmapError(_reconstruction_failure_message(distorted_sparse_dir / "0", image_count, logs))

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


def _run_best_mapper(database_path: Path, images_dir: Path, sparse_dir: Path, logs: list[str], image_count: int) -> Path | None:
    profiles = [
        {
            "name": "strict",
            "min_num_matches": "15",
            "init_min_num_inliers": "80",
            "init_max_error": "4",
            "abs_pose_min_num_inliers": "30",
            "abs_pose_max_error": "8",
            "filter_max_reproj_error": "4",
        },
        {
            "name": "balanced",
            "min_num_matches": "10",
            "init_min_num_inliers": "50",
            "init_max_error": "6",
            "abs_pose_min_num_inliers": "20",
            "abs_pose_max_error": "10",
            "filter_max_reproj_error": "6",
        },
        {
            "name": "fallback",
            "min_num_matches": "6",
            "init_min_num_inliers": "25",
            "init_max_error": "8",
            "abs_pose_min_num_inliers": "12",
            "abs_pose_max_error": "12",
            "filter_max_reproj_error": "8",
        },
    ]
    min_registered = max(24, min(image_count, int(image_count * 0.35)))
    for profile in profiles:
        _reset_sparse_dir(sparse_dir)
        print(f"COLMAP mapper profile={profile['name']}", flush=True)
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
                profile["min_num_matches"],
                "--Mapper.init_min_num_inliers",
                profile["init_min_num_inliers"],
                "--Mapper.init_max_error",
                profile["init_max_error"],
                "--Mapper.abs_pose_min_num_inliers",
                profile["abs_pose_min_num_inliers"],
                "--Mapper.abs_pose_max_error",
                profile["abs_pose_max_error"],
                "--Mapper.filter_max_reproj_error",
                profile["filter_max_reproj_error"],
                "--Mapper.ba_global_function_tolerance",
                "0.000001",
            ],
            logs,
            check=False,
        )
        model = _best_sparse_model(sparse_dir, logs)
        if model is None:
            continue
        registered, points = _model_stats(model, logs)
        print(f"COLMAP model profile={profile['name']} registered_images={registered} points={points}", flush=True)
        if registered >= min_registered:
            return model

    return None


def _run(command: list[str], logs: list[str], check: bool = True) -> int:
    print(f"Running command: {' '.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True, env=_colmap_env())
    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    if output:
        logs.append(f"$ {' '.join(command)}\n{output}")
        print(_tail([output], max_chars=3000), flush=True)
    if check and result.returncode != 0:
        raise ColmapError(f"COLMAP command failed with exit code {result.returncode}: {' '.join(command)}\n{_tail(logs)}")
    return result.returncode


def _colmap_env() -> dict[str, str]:
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
    Path(env["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _normalize_sparse_layout(dataset_dir: Path) -> None:
    sparse_dir = dataset_dir / "sparse"
    sparse_zero_dir = sparse_dir / "0"
    if not _has_sparse_model(sparse_zero_dir):
        source_model = _best_sparse_model(sparse_dir, [])
        if source_model and source_model != sparse_zero_dir:
            sparse_zero_dir.mkdir(parents=True, exist_ok=True)
            for name in ["cameras.bin", "images.bin", "points3D.bin"]:
                source = source_model / name
                if source.exists():
                    shutil.copy2(str(source), str(sparse_zero_dir / name))
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


def _best_sparse_model(sparse_dir: Path, logs: list[str]) -> Path | None:
    candidates = sorted(path for path in sparse_dir.iterdir() if path.is_dir() and _has_sparse_model(path))
    if not candidates:
        return None
    scored = [(_model_stats(candidate, logs), candidate) for candidate in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _model_stats(sparse_model_dir: Path, logs: list[str]) -> tuple[int, int]:
    try:
        result = subprocess.run(
            ["colmap", "model_analyzer", "--path", str(sparse_model_dir)],
            capture_output=True,
            text=True,
            env=_colmap_env(),
            check=False,
        )
        output = "\n".join(part for part in [result.stdout, result.stderr] if part)
        if output:
            logs.append(f"$ colmap model_analyzer --path {sparse_model_dir}\n{output}")
        registered = _first_int(output, [r"Registered images:\s*(\d+)", r"Images:\s*(\d+)"])
        points = _first_int(output, [r"Points:\s*(\d+)", r"Points3D:\s*(\d+)"])
        return registered, points
    except Exception:
        return (0, 0)


def _first_int(text: str, patterns: list[str]) -> int:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return 0


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

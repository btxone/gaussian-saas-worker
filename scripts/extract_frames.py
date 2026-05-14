import subprocess
from pathlib import Path
import math


def extract_frames(normalized_video: Path, frames_dir: Path, settings: dict) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    fps = max(float(settings.get("fps", 10)), 8.0)
    max_frames = _frame_budget(normalized_video, fps, settings)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(normalized_video),
        "-vf",
        f"fps={fps},scale='min(1920,iw)':-2",
        str(frames_dir / "frame_%05d.jpg"),
    ]
    subprocess.run(command, check=True)

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frames) <= max_frames:
        return len(frames)

    keep_indexes = {round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)}
    for index, frame in enumerate(frames):
        if index not in keep_indexes:
            frame.unlink()
    return max_frames


def _frame_budget(normalized_video: Path, fps: float, settings: dict) -> int:
    max_frames = max(int(settings.get("max_frames", 4000)), 1600)
    duration_seconds = _video_duration_seconds(normalized_video)
    if duration_seconds <= 0:
        return max_frames

    expected_frames = math.ceil(duration_seconds * fps)
    return min(max(expected_frames, 1), max_frames)


def _video_duration_seconds(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0

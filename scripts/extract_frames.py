import subprocess
from pathlib import Path


def extract_frames(normalized_video: Path, frames_dir: Path, settings: dict) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    fps = settings.get("fps", 2)
    max_frames = settings.get("max_frames", 600)
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


import subprocess
from pathlib import Path


def normalize_video(input_path: Path, output_path: Path, settings: dict) -> Path:
    max_width = settings.get("max_width", 1920)
    fps = settings.get("fps", 30)
    crf = settings.get("crf", 20)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-vf",
        f"scale='if(gt(iw,{max_width}),{max_width},iw)':'-2',fps={fps}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


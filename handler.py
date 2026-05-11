from pathlib import Path
from tempfile import TemporaryDirectory

import runpod

from scripts.convert_outputs import convert_outputs
from scripts.extract_frames import extract_frames
from scripts.normalize_video import normalize_video
from scripts.run_colmap import run_colmap
from scripts.run_training import run_training
from scripts.storage import download_file, upload_outputs


def handler(job):
    job_input = job["input"]
    project_id = job_input["project_id"]
    input_video_url = job_input["input_video_url"]
    output_prefix = job_input["output_prefix"]
    settings = job_input["settings"]

    try:
        with TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            original = workdir / "original_video.mp4"
            normalized = workdir / "normalized.mp4"
            dataset_dir = workdir / "dataset"
            frames_dir = dataset_dir / "input"
            model_dir = workdir / "model"
            export_dir = workdir / "exports"

            download_file(input_video_url, original)
            normalize_video(original, normalized, settings["normalize"])
            frames_used = extract_frames(normalized, frames_dir, settings["frames"])
            thumbnail_path = next(iter(sorted(frames_dir.glob("*.jpg"))), None)
            run_colmap(dataset_dir)
            ply_path = run_training(dataset_dir, model_dir, settings["training"])
            outputs = convert_outputs(ply_path, export_dir, settings.get("exports", ["ply"]))
            if thumbnail_path:
                thumbnail_target = export_dir / "thumbnail.jpg"
                thumbnail_target.write_bytes(thumbnail_path.read_bytes())
                outputs["thumbnail"] = thumbnail_target
            uploaded = upload_outputs(outputs, output_prefix)

        return {
            "status": "completed",
            "project_id": project_id,
            "outputs": uploaded,
            "metadata": {"frames_used": frames_used},
        }
    except Exception as exc:
        return {
            "status": "failed",
            "project_id": project_id,
            "error": {"code": "PROCESSING_FAILED", "message": str(exc)},
        }


runpod.serverless.start({"handler": handler})

# RunPod Worker Setup

This MVP uses the official Graphdeco 3D Gaussian Splatting implementation inside a RunPod Serverless worker.

## Why This Pipeline

- FFmpeg normalizes videos and extracts stable frames.
- COLMAP estimates cameras with CPU SIFT, phone-friendly OPENCV calibration, high-overlap sequential matching, and mapper quality profiles.
- Graphdeco `train.py` trains the Gaussian Splatting scene.
- PlayCanvas `splat-transform` converts the trained PLY to `.sog` and `.compressed.ply` for SuperSplat viewing.
- The worker uploads generated outputs back to S3-compatible storage.

The backend sends an async RunPod job and later syncs the job result with:

```http
POST /projects/{project_id}/jobs/{job_id}/sync
```

The frontend calls that endpoint automatically while a job is active.

## Build The Docker Image

From the repository root:

```powershell
docker build -t gaussian-saas-worker:latest .
```

Tag and push to your registry:

```powershell
docker tag gaussian-saas-worker:latest your-registry/gaussian-saas-worker:latest
docker push your-registry/gaussian-saas-worker:latest
```

Use that pushed image in a RunPod Serverless endpoint.

## RunPod Environment Variables

Set these variables on the RunPod endpoint:

```env
S3_ENDPOINT=
S3_BUCKET=
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_REGION=us-east-1
S3_KEY_PREFIX=gauss-saas
```

For AWS S3, `S3_ENDPOINT` can be empty. For Cloudflare R2, Backblaze B2, MinIO, or another S3-compatible provider, set the provider endpoint.

The worker also accepts AWS-style aliases (`AWS_S3_BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`) for easier RunPod setup.

## Backend Environment Variables

Set these in `app/backend/.env`:

```env
STORAGE_BACKEND=s3
S3_ENDPOINT=
S3_BUCKET=
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_REGION=us-east-1
S3_KEY_PREFIX=gauss-saas
RUNPOD_API_KEY=
RUNPOD_ENDPOINT_ID=
PUBLIC_APP_URL=http://localhost:3000
BACKEND_PUBLIC_URL=http://localhost:8000
```

## Worker Input Contract

The backend sends:

```json
{
  "input": {
    "project_id": "uuid",
    "input_video_url": "signed-download-url",
    "output_bucket": "bucket-name",
    "output_prefix": "gauss-saas/projects/uuid/output",
    "settings": {
      "normalize": {
        "max_width": 1920,
        "fps": 30,
        "crf": 20
      },
      "frames": {
        "fps": 10,
        "max_frames": 1600
      },
      "training": {
        "quality": "preview",
        "iterations": 30000
      },
      "exports": ["ply", "compressed_ply", "sog", "viewer_html"]
    }
  }
}
```

## Worker Output Contract

On success:

```json
{
  "status": "completed",
  "project_id": "uuid",
  "outputs": {
    "ply": "gauss-saas/projects/uuid/output/model.ply",
    "compressed_ply": "gauss-saas/projects/uuid/output/model.compressed.ply",
    "sog": "gauss-saas/projects/uuid/output/model.sog",
    "viewer_html": "gauss-saas/projects/uuid/output/viewer.html",
    "thumbnail": "gauss-saas/projects/uuid/output/thumbnail.jpg"
  },
  "metadata": {
    "frames_used": 420,
    "conversion_errors": {}
  }
}
```

On failure:

```json
{
  "status": "failed",
  "project_id": "uuid",
  "error": {
    "code": "PROCESSING_FAILED",
    "message": "..."
  }
}
```

## Notes

- The first Docker build is heavy because it compiles Gaussian Splatting CUDA submodules.
- Keep RunPod disk size generous enough for frames, COLMAP output, and model checkpoints.
- For higher quality, increase `training.iterations` and `frames.max_frames`, but expect longer GPU time.
- The worker enforces a practical quality floor: at least 8 fps / 1600 frame budget and 30000 training iterations. COLMAP must register enough frames, otherwise the job fails with diagnostics instead of returning a very poor splat.
- `.sog` is the preferred PlayCanvas/SuperSplat delivery format. `.compressed.ply` and `.ply` are kept as fallbacks for debugging and download.
- The Docker image installs Node.js 20, Vulkan runtime libraries, and `@playcanvas/splat-transform` so RunPod can emit the viewer-friendly formats inside the same job. SOG export tries GPU first and falls back to CPU for reliability.

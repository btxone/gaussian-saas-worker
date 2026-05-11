# RunPod Worker Setup

This MVP uses the official Graphdeco 3D Gaussian Splatting implementation inside a RunPod Serverless worker.

## Why This Pipeline

- FFmpeg normalizes videos and extracts stable frames.
- COLMAP estimates cameras using the official Graphdeco `convert.py` flow.
- Graphdeco `train.py` trains the Gaussian Splatting scene.
- The worker uploads generated outputs back to S3-compatible storage.

The backend sends an async RunPod job and later syncs the job result with:

```http
POST /projects/{project_id}/jobs/{job_id}/sync
```

The frontend calls that endpoint automatically while a job is active.

## Build The Docker Image

From the repository root:

```powershell
docker build -t gaussian-saas-worker:latest app/runpod-worker
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
```

For AWS S3, `S3_ENDPOINT` can be empty. For Cloudflare R2, Backblaze B2, MinIO, or another S3-compatible provider, set the provider endpoint.

## Backend Environment Variables

Set these in `app/backend/.env`:

```env
STORAGE_BACKEND=s3
S3_ENDPOINT=
S3_BUCKET=
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_REGION=us-east-1
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
    "output_prefix": "projects/uuid/output",
    "settings": {
      "normalize": {
        "max_width": 1920,
        "fps": 30,
        "crf": 20
      },
      "frames": {
        "fps": 2,
        "max_frames": 600
      },
      "training": {
        "quality": "preview",
        "iterations": 7000
      },
      "exports": ["ply"]
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
    "ply": "projects/uuid/output/model.ply",
    "thumbnail": "projects/uuid/output/thumbnail.jpg"
  },
  "metadata": {
    "frames_used": 420
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
- `.splat` or `.spz` export should be added only with a real converter. The MVP currently exports high-quality `.ply`, which the viewer attempts to load first.

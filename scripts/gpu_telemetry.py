import json
import subprocess


def gpu_telemetry(stage: str) -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"stage": stage, "available": False, "error": result.stderr.strip()[:500]}

    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        gpus.append(
            {
                "name": parts[0],
                "memory_used_mib": _int_or_none(parts[1]),
                "memory_total_mib": _int_or_none(parts[2]),
                "gpu_utilization_percent": _int_or_none(parts[3]),
                "memory_utilization_percent": _int_or_none(parts[4]),
                "temperature_c": _int_or_none(parts[5]),
                "pstate": parts[6],
            }
        )
    return {"stage": stage, "available": True, "gpus": gpus}


def print_gpu_telemetry(stage: str) -> None:
    print(json.dumps({"message": "gpu_telemetry", **gpu_telemetry(stage)}, default=str), flush=True)


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None

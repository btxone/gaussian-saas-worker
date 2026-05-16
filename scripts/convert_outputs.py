from pathlib import Path
import os
import shutil
import subprocess

CONVERSION_TIMEOUT_SECONDS = int(os.getenv("SPLAT_TRANSFORM_TIMEOUT_SECONDS", "1800"))


def convert_outputs(ply_path: Path, export_dir: Path, exports: list[str]) -> tuple[dict[str, Path], dict[str, str]]:
    export_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    conversion_errors: dict[str, str] = {}
    requested_exports = set(exports or ["ply"])

    if "ply" in requested_exports:
        target = export_dir / "model.ply"
        shutil.copy2(ply_path, target)
        outputs["ply"] = target

    conversion_targets = {
        "compressed_ply": (
            export_dir / "model.compressed.ply",
            [["splat-transform", "-w", str(ply_path), str(export_dir / "model.compressed.ply")]],
        ),
        "sog": (
            export_dir / "model.sog",
            [
                ["splat-transform", "-w", str(ply_path), str(export_dir / "model.sog")],
                ["splat-transform", "-w", "-g", "cpu", str(ply_path), str(export_dir / "model.sog")],
            ],
        ),
        "viewer_html": (
            export_dir / "viewer.html",
            [
                ["splat-transform", "-w", str(ply_path), str(export_dir / "viewer.html")],
                ["splat-transform", "-w", "-g", "cpu", str(ply_path), str(export_dir / "viewer.html")],
            ],
        ),
    }

    for export_name, (target, commands) in conversion_targets.items():
        if export_name not in requested_exports:
            continue
        try:
            print(f"Converting {export_name} to {target}", flush=True)
            _run_any(commands)
            outputs[export_name] = target
            print(f"Converted {export_name} to {target}", flush=True)
        except Exception as exc:
            conversion_errors[export_name] = str(exc)
            print(f"Conversion {export_name} failed: {exc}", flush=True)

    return outputs, conversion_errors


def _run_any(commands: list[list[str]]) -> None:
    errors: list[str] = []
    for command in commands:
        try:
            _run(command)
            return
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("\n\n".join(errors))


def _run(command: list[str]) -> None:
    print(f"Running conversion command: {' '.join(command)}", flush=True)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=CONVERSION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        logs = "\n".join(part for part in [exc.stdout, exc.stderr] if part)
        raise RuntimeError(
            f"{' '.join(command)} timed out after {CONVERSION_TIMEOUT_SECONDS}s\n{_tail(logs)}"
        ) from exc
    if result.returncode != 0:
        logs = "\n".join(part for part in [result.stdout, result.stderr] if part)
        raise RuntimeError(f"{' '.join(command)} failed with exit code {result.returncode}\n{_tail(logs)}")


def _tail(value: str, max_chars: int = 2000) -> str:
    clean_value = value.strip()
    return clean_value if len(clean_value) <= max_chars else clean_value[-max_chars:]

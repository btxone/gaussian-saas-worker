from pathlib import Path


def convert_outputs(ply_path: Path, export_dir: Path, exports: list[str]) -> dict[str, Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    if "ply" in exports:
        target = export_dir / "model.ply"
        target.write_bytes(ply_path.read_bytes())
        outputs["ply"] = target

    return outputs

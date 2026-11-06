from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_ISOMETRIC = Path(r"C:\PDC_Training\training\isometric\images\train")
DEFAULT_TOP = Path(r"C:\PDC_Training\training\top\images\train")
DEFAULT_OUTPUT = Path("work/crop_test_batch")

VIEWS = (
    ("isometric", 0, DEFAULT_ISOMETRIC),
    ("top", 1, DEFAULT_TOP),
)


def collect_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    return sorted(
        (path for path in folder.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )


def prepare_batch(
    isometric_dir: Path,
    top_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict:
    sources = (
        ("isometric", 0, isometric_dir),
        ("top", 1, top_dir),
    )
    output = output_dir.resolve()
    source_paths = [folder.resolve() for _, _, folder in sources]

    if any(output == source or output in source.parents for source in source_paths):
        raise ValueError("Output directory cannot be a source folder or its parent.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Choose another path or use --overwrite."
            )
        if any(source == output or source in output.parents for source in source_paths):
            raise ValueError("Refusing to overwrite a directory containing source images.")
        shutil.rmtree(output)

    records = []
    copied_counts = {}
    seen_names = set()
    output.mkdir(parents=True)

    try:
        for view, angle_index, source_dir in sources:
            images = collect_images(source_dir)
            copied_counts[view] = len(images)
            for source in images:
                key = source.name.casefold()
                if key in seen_names:
                    raise ValueError(
                        f"Duplicate filename across source folders: {source.name}"
                    )
                seen_names.add(key)
                shutil.copy2(source, output / source.name)
                records.append({
                    "file_path": source.name,
                    "angle_index": angle_index,
                })

        manifest_path = output / "manifest.json"
        manifest_path.write_text(
            json.dumps(records, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "output": str(output),
            "manifest": str(manifest_path),
            "counts": copied_counts,
            "total": len(records),
        }
        (output / "batch_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a disposable crop-test batch from the Top and Isometric "
            "training image folders. Source images are copied, never modified."
        )
    )
    parser.add_argument("--isometric", type=Path, default=DEFAULT_ISOMETRIC)
    parser.add_argument("--top", type=Path, default=DEFAULT_TOP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output folder if it already exists (never source folders).",
    )
    args = parser.parse_args()

    summary = prepare_batch(
        args.isometric, args.top, args.output, overwrite=args.overwrite
    )
    print("Crop-test batch created.")
    print(f"Output: {summary['output']}")
    print(f"Manifest: {summary['manifest']}")
    print(f"Isometric (angle_index 0): {summary['counts']['isometric']}")
    print(f"Top (angle_index 1): {summary['counts']['top']}")
    print(f"Total copied: {summary['total']}")
    print("Source images were copied; originals were not modified.")


if __name__ == "__main__":
    main()

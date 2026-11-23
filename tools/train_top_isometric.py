from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "training_presentation_exif_normalized"


def train_view(view: str, args) -> None:
    from ultralytics import YOLO

    data_yaml = args.data_root / view / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}. Prepare and review the derived dataset first."
        )
    model = YOLO(args.model)
    run_name = "top_view" if view == "top" else "isometric_view"
    print(f"Training {view} detector")
    print(f"  data: {data_yaml}")
    print(f"  pretrained checkpoint: {args.model}")
    print(f"  output: {ROOT / 'models' / run_name}")

    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        pretrained=True,
        project=str(ROOT / "models"),
        name=run_name,
        exist_ok=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train the independent Top and Isometric YOLO object detectors."
    )
    parser.add_argument("--view", choices=("top", "isometric", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained Ultralytics detection checkpoint.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default="0", help="GPU index such as 0, or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="Default 0 is Windows-friendly.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    views = ("top", "isometric") if args.view == "all" else (args.view,)
    for view in views:
        train_view(view, args)


if __name__ == "__main__":
    main()

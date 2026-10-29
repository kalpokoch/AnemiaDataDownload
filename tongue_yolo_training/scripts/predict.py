from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # tongue_yolo_training/
DATA_YAML = ROOT / "data.yaml"
WEIGHTS = ROOT / "runs" / "segment" / "tongue_back_yolov8n" / "weights" / "best.pt"


def main():
    model = YOLO(str(WEIGHTS))

    model.val(data=str(DATA_YAML), split="test")

    sample_images = sorted((ROOT / "test" / "images").glob("*.jpg"))[:1]
    model.predict(
        source=[str(p) for p in sample_images],
        save=True,
        project="runs",
        name="tongue_back_seg_predict",
    )


if __name__ == "__main__":
    main()

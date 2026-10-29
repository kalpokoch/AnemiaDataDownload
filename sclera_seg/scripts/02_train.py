"""Train a YOLOv8n-seg model on the Sclera instance-segmentation dataset."""
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = "/workspace/Sclera.v1i.yolov8/data.yaml"
PROJECT_DIR = "/workspace/sclera_seg/runs/segment"
RUN_NAME = "sclera_yolov8n"


def main():
    model = YOLO("/workspace/sclera_seg/yolov8n-seg.pt")

    model.train(
        data=DATA_YAML,
        epochs=150,
        patience=30,
        imgsz=640,
        batch=16,
        device=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        seed=0,
        deterministic=True,
        val=True,
        plots=True,
        save=True,
        workers=8,
        optimizer="auto",
        lr0=0.01,
        cos_lr=True,
        # Roboflow already applied flip/rotation/brightness/exposure augmentation
        # offline (3x per source image) -- keep online augmentation light to avoid
        # compounding on top of that.
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=0.0,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.2,
    )

    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    print("Training complete. Best weights at:", best_weights)


if __name__ == "__main__":
    main()

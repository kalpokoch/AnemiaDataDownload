"""Evaluate the trained YOLOv8n-seg model on the held-out test split and build
GT-vs-prediction comparison visualizations + a metrics summary.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

BEST_WEIGHTS = "/workspace/palm_seg/runs/segment/palm_yolov8n/weights/best.pt"
DATA_YAML = "/workspace/PalmDetection.v1i.yolov8/data.yaml"
TEST_IMG_DIR = Path("/workspace/PalmDetection.v1i.yolov8/test/images")
TEST_LBL_DIR = Path("/workspace/PalmDetection.v1i.yolov8/test/labels")
OUT_DIR = Path("/workspace/palm_seg/outputs/predictions")
CLASS_NAMES = ["PalmDetection"]
N_GRID_IMAGES = 12


def load_polygon_label(label_path: Path):
    objects = []
    if not label_path.exists():
        return objects
    text = label_path.read_text().strip()
    if not text:
        return objects
    for line in text.splitlines():
        vals = line.split()
        cls_id = int(vals[0])
        coords = list(map(float, vals[1:]))
        points = list(zip(coords[0::2], coords[1::2]))
        objects.append((cls_id, points))
    return objects


def polygon_pixel_mask(points, img_w, img_h):
    pts_px = np.array([[x * img_w, y * img_h] for x, y in points], dtype=np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_px], 1)
    return mask


def overlay_gt_on_image(img, objects, color=(0, 255, 0)):
    overlay = img.copy()
    h, w = img.shape[:2]
    for cls_id, points in objects:
        mask = polygon_pixel_mask(points, w, h)
        colored = np.zeros_like(img)
        colored[mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)
        pts_px = np.array([[x * w, y * h] for x, y in points], dtype=np.int32)
        cv2.polylines(overlay, [pts_px], isClosed=True, color=color, thickness=2)
        x_min, y_min = pts_px.min(axis=0)
        cv2.putText(overlay, CLASS_NAMES[cls_id], (int(x_min), max(0, int(y_min) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return overlay


def overlay_pred_on_image(img, result, color=(255, 0, 0)):
    overlay = img.copy()
    if result.masks is None:
        return overlay
    for mask_xy, conf in zip(result.masks.xy, result.boxes.conf):
        pts = mask_xy.astype(np.int32)
        colored = np.zeros_like(overlay)
        cv2.fillPoly(colored, [pts], color)
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)
        cv2.polylines(overlay, [pts], True, color, 2)
        x, y = pts.min(axis=0)
        cv2.putText(overlay, f"{conf:.2f}", (int(x), max(0, int(y) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return overlay


def run_formal_validation(model):
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        plots=True,
        project=str(OUT_DIR),
        name="test_val",
        exist_ok=True,
    )
    return {
        "box_mAP50": float(metrics.box.map50),
        "box_mAP50-95": float(metrics.box.map),
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "mask_mAP50": float(metrics.seg.map50),
        "mask_mAP50-95": float(metrics.seg.map),
        "mask_precision": float(metrics.seg.mp),
        "mask_recall": float(metrics.seg.mr),
    }


def build_prediction_grid(model, n=N_GRID_IMAGES):
    test_images = sorted(TEST_IMG_DIR.glob("*.jpg"))[:n]
    results = model.predict(source=[str(p) for p in test_images], imgsz=640,
                             conf=0.25, iou=0.6, save=False, verbose=False)

    fig, axes = plt.subplots(len(test_images), 2, figsize=(8, 4 * len(test_images)))
    if len(test_images) == 1:
        axes = axes.reshape(1, 2)
    for i, (img_path, pred) in enumerate(zip(test_images, results)):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gt_objects = load_polygon_label(TEST_LBL_DIR / (img_path.stem + ".txt"))
        gt_panel = overlay_gt_on_image(img, gt_objects)
        pred_panel = overlay_pred_on_image(img, pred)

        axes[i, 0].imshow(gt_panel)
        axes[i, 0].set_title(f"GT: {img_path.name}", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(pred_panel)
        axes[i, 1].set_title("Prediction", fontsize=8)
        axes[i, 1].axis("off")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / "test_predictions_grid.png", dpi=150)
    plt.close(fig)


def plot_metrics_summary(metrics_dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics_dict])
    df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    keys = list(metrics_dict.keys())
    values = [metrics_dict[k] for k in keys]
    ax.bar(keys, values)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Test Set Metrics Summary")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "metrics_summary.png", dpi=150)
    plt.close(fig)


def main():
    model = YOLO(BEST_WEIGHTS)
    metrics_dict = run_formal_validation(model)
    plot_metrics_summary(metrics_dict)
    build_prediction_grid(model)
    print("Test set metrics:")
    for k, v in metrics_dict.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()

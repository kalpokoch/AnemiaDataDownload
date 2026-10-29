"""Dataset sanity-check and ground-truth visualization for the TongueFront YOLOv8-seg dataset.

Pure opencv/numpy/matplotlib/shapely - no ultralytics dependency, so it can run
independently of the training setup.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon

DATA_ROOT = Path("/workspace/TongueFront.v1i.yolov8")
OUT_DIR = Path("/workspace/tongue_front_seg/outputs/eda")
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = ["TongueFront"]

random.seed(0)


def load_polygon_label(label_path: Path):
    """Parse a YOLO-seg label file -> list of (class_id, [(x, y), ...]) in normalized coords."""
    objects = []
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


def check_polygon_validity(points):
    unique_points = list(dict.fromkeys(points))
    out_of_bounds = any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in points)
    degenerate = len(unique_points) < 3
    self_intersecting = False
    if not degenerate:
        poly = Polygon(unique_points)
        degenerate = degenerate or poly.area < 1e-6
        self_intersecting = not poly.is_valid
    return {
        "n_unique_vertices": len(unique_points),
        "out_of_bounds": out_of_bounds,
        "degenerate": degenerate,
        "self_intersecting": self_intersecting,
    }


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


def strip_roboflow_suffix(filename: str) -> str:
    """'<source>_jpg.rf.<hash>.jpg' -> '<source>' for leakage checking across splits."""
    stem = Path(filename).stem
    if "_jpg.rf." in stem:
        return stem.split("_jpg.rf.")[0]
    return stem


def summarize_split(split: str):
    img_dir = DATA_ROOT / split / "images"
    lbl_dir = DATA_ROOT / split / "labels"
    rows = []
    source_ids = set()
    for img_path in sorted(img_dir.glob("*.jpg")):
        source_ids.add(strip_roboflow_suffix(img_path.name))
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        objects = load_polygon_label(lbl_path) if lbl_path.exists() else []
        for cls_id, points in objects:
            mask = polygon_pixel_mask(points, w, h)
            mask_area_pct = mask.sum() / (w * h) * 100
            xs, ys = zip(*points)
            bbox_w, bbox_h = (max(xs) - min(xs)) * w, (max(ys) - min(ys)) * h
            aspect_ratio = bbox_w / bbox_h if bbox_h else float("nan")
            validity = check_polygon_validity(points)
            rows.append({
                "split": split,
                "image": img_path.name,
                "class_id": cls_id,
                "mask_area_pct": mask_area_pct,
                "bbox_aspect_ratio": aspect_ratio,
                "img_w": w,
                "img_h": h,
                **validity,
            })
    df = pd.DataFrame(rows)
    return df, source_ids


def plot_sample_grid(split: str, n: int = 6):
    img_dir = DATA_ROOT / split / "images"
    lbl_dir = DATA_ROOT / split / "labels"
    img_paths = sorted(img_dir.glob("*.jpg"))
    sample = random.sample(img_paths, min(n, len(img_paths)))

    ncols = 3
    nrows = (len(sample) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, img_path in zip(axes, sample):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        objects = load_polygon_label(lbl_path) if lbl_path.exists() else []
        overlay = overlay_gt_on_image(img, objects)
        ax.imshow(overlay)
        ax.set_title(img_path.name, fontsize=7)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")
    fig.suptitle(f"Ground truth samples - {split}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"sample_overlays_{split}.png", dpi=150)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="mask_area_pct", hue="split", bins=20, kde=True, ax=ax)
    ax.set_title("Mask area as % of image, by split")
    ax.set_xlabel("Mask area (% of image)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mask_area_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="bbox_aspect_ratio", hue="split", bins=20, kde=True, ax=ax)
    ax.set_title("Bounding-box aspect ratio (w/h), by split")
    ax.set_xlabel("Aspect ratio")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "aspect_ratio_distribution.png", dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    split_summaries = []
    source_ids_by_split = {}

    for split in SPLITS:
        df, source_ids = summarize_split(split)
        all_dfs.append(df)
        source_ids_by_split[split] = source_ids
        plot_sample_grid(split)

        n_images = df["image"].nunique()
        n_objects = len(df)
        n_invalid = int((df["out_of_bounds"] | df["degenerate"] | df["self_intersecting"]).sum())
        split_summaries.append({
            "split": split,
            "n_images": n_images,
            "n_objects": n_objects,
            "objects_per_image": n_objects / n_images if n_images else float("nan"),
            "n_invalid_polygons": n_invalid,
            "n_unique_source_images": len(source_ids),
        })

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(OUT_DIR / "polygon_validity_report.csv", index=False)

    summary_df = pd.DataFrame(split_summaries)
    summary_df.to_csv(OUT_DIR / "split_summary.csv", index=False)

    plot_distributions(full_df)

    print("=== Split summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Source-image leakage check (stripped Roboflow hash suffix) ===")
    train_ids, valid_ids, test_ids = (source_ids_by_split["train"],
                                       source_ids_by_split["valid"],
                                       source_ids_by_split["test"])
    overlap_train_valid = train_ids & valid_ids
    overlap_train_test = train_ids & test_ids
    overlap_valid_test = valid_ids & test_ids
    print(f"train/valid overlap: {len(overlap_train_valid)} source images")
    print(f"train/test overlap:  {len(overlap_train_test)} source images")
    print(f"valid/test overlap:  {len(overlap_valid_test)} source images")
    if overlap_train_valid or overlap_train_test or overlap_valid_test:
        print("WARNING: source-image leakage detected across splits!")
    else:
        print("OK: no source-image leakage across splits.")

    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()

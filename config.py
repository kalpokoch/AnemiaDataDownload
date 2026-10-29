import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GOOGLE_SHEET_URL = os.environ.get("GOOGLE_SHEET_URL", "").strip()

# Temporary override: when set, the app loads data from this local Excel file
# instead of fetching GOOGLE_SHEET_URL. Unset (or leave blank) to go back to the sheet.
LOCAL_SHEET_PATH = os.environ.get("LOCAL_SHEET_PATH", "").strip()
NAME_COL = os.environ.get("NAME_COL", "Name")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "downloaded_data"))
MANIFEST_PATH = OUTPUT_DIR / "sheet_with_image_status.csv"
DOWNLOAD_WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", "4"))

# module key -> sheet column header
DEFAULT_IMAGE_COLS = {
    "lower_eye_lid": "Pic 1 (lower eye lid)",
    "tongue_front": "Pic 2 (tongue)-front",
    "fingertips": "Pic 4 Fingertips",
    "sclera": "Pic 3 (Sclera)",
    "palm": "Palm",
    "tongue_back": "Tongue back",
}

_image_cols_override = os.environ.get("IMAGE_COLS_JSON", "").strip()
IMAGE_COLS = json.loads(_image_cols_override) if _image_cols_override else DEFAULT_IMAGE_COLS

# module key -> trained YOLO segmentation weights used to crop downloaded photos.
# Only modules with a model here get cropped; add more entries as models are trained.
YOLO_MODEL_PATHS = {
    "lower_eye_lid": Path(__file__).resolve().parent / "yolo_training" / "runs" / "segment" / "conjunctiva_yolov8n" / "weights" / "best.pt",
    "tongue_back": Path(__file__).resolve().parent / "tongue_yolo_training" / "runs" / "segment" / "tongue_back_yolov8n" / "weights" / "best.pt",
    "tongue_front": Path(__file__).resolve().parent / "tongue_front_seg" / "runs" / "segment" / "tongue_front_yolov8n" / "weights" / "best.pt",
    "sclera": Path(__file__).resolve().parent / "sclera_seg" / "runs" / "segment" / "sclera_yolov8n" / "weights" / "best.pt",
    "palm": Path(__file__).resolve().parent / "palm_seg" / "runs" / "segment" / "palm_yolov8n" / "weights" / "best.pt",
}

# Modules in this set use polygon-masked cropping (background zeroed, tight bbox crop)
# rather than plain bounding-box crop.
YOLO_MASK_MODULES = {"tongue_back", "tongue_front", "sclera", "palm"}

YOLO_CONF_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))

# Path to the dataset.csv produced by "Build dataset CSV" in app.py.
DATASET_PATH = OUTPUT_DIR / "dataset.csv"

# Where the "Clean Dataset" page writes its filtered/cleaned output.
CLEAN_DATASET_PATH = OUTPUT_DIR / "cleaned_dataset.csv"

# Where the "Clean Dataset" page writes the label-encoding map for categorical columns.
ENCODING_MAP_PATH = OUTPUT_DIR / "encoding_map.json"

# Persisted raw-value -> canonical-value corrections made in the Clean Dataset page.
# Kept at repo root (unlike OUTPUT_DIR) so the mappings survive a downloaded_data wipe.
CORRECTIONS_PATH = Path(os.environ.get("CORRECTIONS_PATH", "corrections.json"))

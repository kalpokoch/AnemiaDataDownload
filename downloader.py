import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

import cropper


def slugify(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r'[\/\\\:\*\?\"\<\>\|]', '-', text)
    text = re.sub(r'\s+', '_', text)
    return text


def extract_gdrive_file_id(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    url = url.strip().strip("'").strip('"')

    m = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)

    m = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)

    m = re.search(r'/d/([a-zA-Z0-9_-]+)/', url)
    if m:
        return m.group(1)
    return None


_IMAGE_MAGIC_PREFIXES = (
    b"\xff\xd8\xff",  # JPEG
    b"\x89PNG\r\n\x1a\n",  # PNG (Drive occasionally re-encodes)
)


def _looks_like_image(path: Path) -> bool:
    """Cheap magic-byte check. Google's confirm-download hop can silently return an
    HTML page (rate-limit wall, sign-in redirect) with a 200 status and no reliable
    Content-Type -- this catches that instead of saving garbage as a .jpg."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return False
    return any(head.startswith(sig) for sig in _IMAGE_MAGIC_PREFIXES)


def _write_response_if_image(response, save_path: Path) -> bool:
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)
    if _looks_like_image(save_path):
        return True
    save_path.unlink(missing_ok=True)
    return False


def _backoff_seconds(attempt: int) -> float:
    # Exponential backoff, capped -- Google's anonymous download endpoint rate-limits in
    # bursts; a short linear backoff (the old 1.5s/3s/4.5s) never outlasts that window.
    return min(3 * (2 ** (attempt - 1)), 30)


def download_from_gdrive(
    file_id: str,
    save_path: Path,
    timeout: int = 30,
    max_retries: int = 5,
    session: Optional[requests.Session] = None,
):
    if save_path.exists() and save_path.stat().st_size > 0:
        return True, "exists"

    if not file_id:
        return False, "missing_file_id"

    requester = session or requests
    base = f"https://drive.google.com/uc?export=download&id={file_id}"

    for attempt in range(1, max_retries + 1):
        try:
            with requester.get(base, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    time.sleep(_backoff_seconds(attempt))
                    continue

                if 'text/html' in r.headers.get('Content-Type', ''):
                    html = r.text
                    m = re.search(r'href="(\/uc\?export=download[^"]*confirm=[^"]+)"', html)
                    if m:
                        confirm_url = "https://drive.google.com" + m.group(1)
                        with requester.get(confirm_url, stream=True, timeout=timeout) as rc:
                            if rc.status_code == 200 and _write_response_if_image(rc, save_path):
                                return True, "ok(confirm)"
                        time.sleep(_backoff_seconds(attempt))
                        continue

                if _write_response_if_image(r, save_path):
                    return True, "ok"
                time.sleep(_backoff_seconds(attempt))
                continue

        except requests.RequestException:
            time.sleep(_backoff_seconds(attempt))
            continue
        except OSError:
            # Transient local file-write failure (e.g. antivirus or a cloud-sync client
            # briefly locking the newly-created file) -- retry like a network hiccup
            # instead of failing this row outright on the first hit.
            time.sleep(_backoff_seconds(attempt))
            continue

    return False, "failed_after_retries"


def _build_session(max_workers: int) -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def sheet_url_to_csv_export(share_url: str) -> str:
    m = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', share_url)
    if not m:
        raise ValueError(f"Could not extract a sheet ID from URL: {share_url}")
    sheet_id = m.group(1)

    m_gid = re.search(r'[?&#]gid=(\d+)', share_url)
    gid = m_gid.group(1) if m_gid else "0"

    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def load_sheet(sheet_url: str) -> pd.DataFrame:
    csv_url = sheet_url_to_csv_export(sheet_url)
    return pd.read_csv(csv_url)


def load_local_sheet(path) -> pd.DataFrame:
    return pd.read_excel(path)


def _classify_link(link) -> str:
    if pd.isna(link) or not str(link).startswith(("http://", "https://")):
        return "missing_link"
    if not extract_gdrive_file_id(str(link)):
        return "bad_url"
    return "has_link"


def module_dir(out_dir: Path, column: str) -> Path:
    return out_dir / slugify(column)


def cropped_dir(out_dir: Path, column: str) -> Path:
    return out_dir / f"{slugify(column)}_cropped"


def failed_crops_dir(out_dir: Path, column: str) -> Path:
    return out_dir / f"{slugify(column)}_failed_crops"


def _expected_filename(person_name, column: str, ridx: int) -> str:
    return f"{slugify(person_name)}__{slugify(column)}__r{ridx + 1}.jpg"


def analyze_links(df: pd.DataFrame, image_cols: dict, out_dir: Path, name_col: str = "Name") -> pd.DataFrame:
    rows = []
    total_rows = len(df)
    for module_key, column in image_cols.items():
        if column not in df.columns:
            rows.append({
                "module": module_key, "column": column, "total_rows": total_rows,
                "downloaded": 0, "cropped": 0, "failed_crop": 0, "pending": 0, "missing_link": total_rows, "bad_url": 0,
            })
            continue

        target_dir = module_dir(out_dir, column)
        crop_dir = cropped_dir(out_dir, column)
        failed_dir = failed_crops_dir(out_dir, column)
        downloaded = cropped = failed_crop = pending = missing_link = bad_url = 0

        for ridx, row in df.iterrows():
            link_class = _classify_link(row[column])
            if link_class == "missing_link":
                missing_link += 1
            elif link_class == "bad_url":
                bad_url += 1
            else:
                person_name = row[name_col] if pd.notna(row.get(name_col)) else f"row{ridx + 1}"
                filename = _expected_filename(person_name, column, ridx)
                if (target_dir / filename).exists():
                    downloaded += 1
                    if (crop_dir / filename).exists():
                        cropped += 1
                    elif (failed_dir / filename).exists():
                        failed_crop += 1
                else:
                    pending += 1

        rows.append({
            "module": module_key,
            "column": column,
            "total_rows": total_rows,
            "downloaded": downloaded,
            "cropped": cropped,
            "failed_crop": failed_crop,
            "pending": pending,
            "missing_link": missing_link,
            "bad_url": bad_url,
        })
    return pd.DataFrame(rows)


def download_module(
    df: pd.DataFrame,
    module_key: str,
    column: str,
    out_dir: Path,
    name_col: str = "Name",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    max_workers: int = 10,
    model_path: Optional[Path] = None,
    crop_conf: float = 0.25,
    crop_masked: bool = False,
) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in sheet")

    target_dir = module_dir(out_dir, column)
    target_dir.mkdir(parents=True, exist_ok=True)

    result_df = df.copy()
    result_df[f"{column}__file"] = ""
    result_df[f"{column}__status"] = ""

    total_rows = len(result_df)
    completed = 0

    def report(status):
        nonlocal completed
        completed += 1
        if progress_cb:
            progress_cb(completed, total_rows, status)

    session = _build_session(max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for ridx, row in result_df.iterrows():
            person_name = row[name_col] if pd.notna(row.get(name_col)) else f"row{ridx + 1}"
            link = row[column]

            link_class = _classify_link(link)
            if link_class in ("missing_link", "bad_url"):
                result_df.at[ridx, f"{column}__status"] = link_class
                report(link_class)
                continue

            file_id = extract_gdrive_file_id(str(link))
            filename = _expected_filename(person_name, column, ridx)
            save_path = target_dir / filename

            future = executor.submit(download_from_gdrive, file_id, save_path, 30, 5, session)
            futures[future] = (ridx, filename)

        for future in as_completed(futures):
            ridx, filename = futures[future]
            try:
                ok, note = future.result()
            except Exception as e:
                ok, note = False, f"exception:{e}"

            if ok:
                status = "exists" if note == "exists" else "ok"
                result_df.at[ridx, f"{column}__file"] = filename
            else:
                status = f"error:{note}"
            result_df.at[ridx, f"{column}__status"] = status
            report(status)

    session.close()
    return result_df


def recrop_existing(
    out_dir: Path,
    column: str,
    model_path: Path,
    conf: float = 0.25,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    masked: bool = False,
    overwrite: bool = False,
) -> dict:
    """Crop the originals in the module folder into <module>_cropped/. Originals are
    never touched; images with no detection are copied to <module>_failed_crops/."""
    target_dir = module_dir(out_dir, column)
    crop_dir = cropped_dir(out_dir, column)
    failed_dir = failed_crops_dir(out_dir, column)
    files = sorted(p for p in target_dir.glob("*.jpg") if p.is_file())

    cropped = failed = skipped = 0
    for i, path in enumerate(files, start=1):
        out_path = crop_dir / path.name
        failed_path = failed_dir / path.name

        if not overwrite and (out_path.exists() or failed_path.exists()):
            skipped += 1
            status = "skipped"
        elif cropper.crop_to_detection(path, out_path, model_path, conf=conf, masked=masked):
            failed_path.unlink(missing_ok=True)
            cropped += 1
            status = "cropped"
        else:
            out_path.unlink(missing_ok=True)
            failed_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, failed_path)
            failed += 1
            status = "no_detection"
        if progress_cb:
            progress_cb(i, len(files), status)

    return {"total": len(files), "cropped": cropped, "failed": failed, "skipped": skipped}


# Sheet column header -> clean output column name for the dataset export.
DATASET_COLUMN_MAP = {
    "Name": "Name",
    "Age (rs)": "Age",
    "Gender (M/F)": "gender",
    "Body weight (Kg)": "weight",
    "Height (cm)": "height",
    "Race (Bengali, Rajbanshi, Rava, Nepali, Bhutia, Marwari, Bihari, Punjabi, South Indian, Others)": "race",
    "Complexion (Dark/Fair)": "complexion",
    "Hb%": "hb%",
}


def build_dataset(
    df: pd.DataFrame,
    image_cols: dict,
    out_dir: Path,
    name_col: str = "Name",
    cropped_modules: Optional[set] = None,
) -> pd.DataFrame:
    """One row per image. img_path is the original download; crop_path is the cropped copy.

    Modules in cropped_modules only get rows that have a crop. Other modules have no
    model, so crop_path just repeats img_path."""
    cropped_modules = cropped_modules or set()
    missing = [c for c in DATASET_COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet is missing expected columns: {missing}")

    demo_cols = list(DATASET_COLUMN_MAP.keys())
    rows = []

    for module_key, column in image_cols.items():
        if column not in df.columns:
            continue

        target_dir = module_dir(out_dir, column)

        for ridx, row in df.iterrows():
            if _classify_link(row[column]) != "has_link":
                continue

            person_name = row[name_col] if pd.notna(row.get(name_col)) else f"row{ridx + 1}"
            filename = _expected_filename(person_name, column, ridx)
            file_path = target_dir / filename
            if not file_path.exists():
                continue

            record = {DATASET_COLUMN_MAP[c]: row[c] for c in demo_cols}
            record["Name"] = person_name
            record["image_type"] = module_key
            record["img_path"] = file_path.relative_to(out_dir).as_posix()
            if module_key in cropped_modules:
                crop_file = cropped_dir(out_dir, column) / filename
                if not crop_file.exists():
                    continue
                record["crop_path"] = crop_file.relative_to(out_dir).as_posix()
            else:
                record["crop_path"] = record["img_path"]
            rows.append(record)

    output_cols = [DATASET_COLUMN_MAP[c] for c in demo_cols] + ["image_type", "img_path", "crop_path"]
    return pd.DataFrame(rows, columns=output_cols)

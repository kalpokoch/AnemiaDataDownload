import json
from pathlib import Path
from typing import Optional

import pandas as pd

# Canonical categories. Any raw value not in this list (after correction) is unresolved.
CATEGORY_CANONICAL = {
    "race": ["Adivashi","Assamese","Bengali", "Rajbanshi", "Rava", "Nepali", "Bhutia", "Marwari", "Bihari", "Punjabi", "South Indian", "Others"],
    "complexion": ["Dark", "Fair", "Medium"],
    "gender": ["M", "F"],
}

# (min, max) plausible ranges used to flag numeric outliers; does not auto-drop rows.
NUMERIC_RANGES = {
    "Age": (1, 120),
    "weight": (2, 300),
    "height": (50, 250),
    "hb%": (2, 25),
}


def load_corrections(path: Path) -> dict:
    data = json.loads(path.read_text()) if path.exists() else {}
    for col in CATEGORY_CANONICAL:
        data.setdefault(col, {})
    return data


def save_corrections(path: Path, corrections: dict) -> None:
    path.write_text(json.dumps(corrections, indent=2, ensure_ascii=False, sort_keys=True))


def _clean_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def suggest_canonical(value: str, canonical: list) -> Optional[str]:
    """Case-insensitive exact match, to pre-fill the obvious casing fixes."""
    lowered = value.lower()
    for c in canonical:
        if c.lower() == lowered:
            return c
    return None


def resolve_category(series: pd.Series, canonical: list, mapping: dict) -> pd.Series:
    canonical_set = set(canonical)

    def resolve(raw):
        v = _clean_str(raw)
        if v in canonical_set:
            return v
        return mapping.get(v, v)

    return series.apply(resolve)


def unresolved_values(series: pd.Series, canonical: list, mapping: dict) -> list:
    """Raw (uncorrected) values that still don't resolve to a canonical category, with row counts."""
    canonical_set = set(canonical)
    raw = series.apply(_clean_str)
    resolved = resolve_category(series, canonical, mapping)
    still_unresolved = ~resolved.isin(canonical_set)
    counts = raw[still_unresolved].value_counts()
    return list(counts.items())


def missing_image_mask(df: pd.DataFrame, image_root: Path) -> pd.Series:
    def check(rel_path):
        v = _clean_str(rel_path)
        if not v:
            return True
        return not (image_root / v).exists()

    # Prefer the cropped image when the dataset has one; fall back to the original.
    path_col = "crop_path" if "crop_path" in df.columns else "img_path"
    return df[path_col].apply(check)


def missing_hb_mask(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["hb%"], errors="coerce").isna()


# Demographic fields that are informational-only: rows are kept even when these are blank.
DEMOGRAPHIC_FIELDS = ["Age", "weight", "height", "gender", "complexion", "race"]
NUMERIC_DEMOGRAPHIC_FIELDS = {"Age", "weight", "height"}


def missing_field_masks(df: pd.DataFrame) -> dict:
    """Per-field blank/NaN masks for DEMOGRAPHIC_FIELDS. Reporting only -- never used to exclude rows."""
    masks = {}
    for col in DEMOGRAPHIC_FIELDS:
        if col in NUMERIC_DEMOGRAPHIC_FIELDS:
            masks[col] = pd.to_numeric(df[col], errors="coerce").isna()
        else:
            masks[col] = df[col].apply(_clean_str) == ""
    return masks


def numeric_outlier_masks(df: pd.DataFrame) -> dict:
    masks = {}
    for col, (lo, hi) in NUMERIC_RANGES.items():
        numeric = pd.to_numeric(df[col], errors="coerce")
        masks[col] = numeric.notna() & ((numeric < lo) | (numeric > hi))
    return masks


def build_report(df: pd.DataFrame, image_root: Path, corrections: dict) -> dict:
    img_missing_mask = missing_image_mask(df, image_root)
    hb_missing_mask = missing_hb_mask(df)

    category_unresolved_masks = {}
    for col, canonical in CATEGORY_CANONICAL.items():
        resolved = resolve_category(df[col], canonical, corrections.get(col, {}))
        category_unresolved_masks[col] = ~resolved.isin(set(canonical))

    outlier_masks = numeric_outlier_masks(df)

    return {
        "total_rows": len(df),
        "img_missing_mask": img_missing_mask,
        "hb_missing_mask": hb_missing_mask,
        "category_unresolved_masks": category_unresolved_masks,
        "outlier_masks": outlier_masks,
        "missing_field_masks": missing_field_masks(df),
    }


def build_encoding_map() -> dict:
    """Fixed label_encoding per category, in CATEGORY_CANONICAL order, for feeding training scripts."""
    return {
        col: {"label_encoding": {value: idx for idx, value in enumerate(canonical)}}
        for col, canonical in CATEGORY_CANONICAL.items()
    }


def save_encoding_map(path: Path) -> None:
    path.write_text(json.dumps(build_encoding_map(), indent=2, ensure_ascii=False))


def build_clean_dataset(df: pd.DataFrame, image_root: Path, corrections: dict, exclude_outliers: bool = False):
    """Returns (clean_df, needs_review_df, summary_dict).

    Rows are hard-excluded only for a missing image or missing/invalid hb% -- nothing in
    this tool can fix those. A row with an unresolved race/complexion/gender value is KEPT
    (its image and hb% are still usable); only the unresolved field(s) are left blank.
    needs_review_df lists those rows for visibility -- map them via the UI above and rebuild
    to fill the blanks in on the next export.
    """
    report = build_report(df, image_root, corrections)

    hard_excluded_mask = report["img_missing_mask"] | report["hb_missing_mask"]

    category_mask = pd.Series(False, index=df.index)
    for mask in report["category_unresolved_masks"].values():
        category_mask = category_mask | mask

    outlier_mask = pd.Series(False, index=df.index)
    for mask in report["outlier_masks"].values():
        outlier_mask = outlier_mask | mask

    needs_review_mask = category_mask & ~hard_excluded_mask

    exclude_mask = hard_excluded_mask
    if exclude_outliers:
        exclude_mask = exclude_mask | outlier_mask

    clean_df = df[~exclude_mask].copy()
    for col, canonical in CATEGORY_CANONICAL.items():
        canonical_set = set(canonical)
        resolved = resolve_category(clean_df[col], canonical, corrections.get(col, {}))
        clean_df[col] = resolved.where(resolved.isin(canonical_set), other=pd.NA)
    clean_df["hb%"] = pd.to_numeric(clean_df["hb%"], errors="coerce")

    needs_review_df = df[needs_review_mask].copy()

    clean_missing_field_masks = missing_field_masks(clean_df)
    incomplete_demographics_mask = pd.Series(False, index=clean_df.index)
    for mask in clean_missing_field_masks.values():
        incomplete_demographics_mask = incomplete_demographics_mask | mask

    summary = {
        "total_rows": report["total_rows"],
        "img_missing": int(report["img_missing_mask"].sum()),
        "hb_missing": int(report["hb_missing_mask"].sum()),
        "category_unresolved": int(category_mask.sum()),
        "needs_review": int(needs_review_mask.sum()),
        "outliers_flagged": int(outlier_mask.sum()),
        "outliers_excluded": exclude_outliers,
        "usable": len(clean_df),
        "complete_demographics": int(len(clean_df) - incomplete_demographics_mask.sum()),
        "modality_breakdown": clean_df["image_type"].value_counts().to_dict(),
    }

    return clean_df, needs_review_df, summary

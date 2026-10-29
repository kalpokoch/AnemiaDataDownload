import json

import pandas as pd
import streamlit as st

import cleaner
import config

st.set_page_config(page_title="Anemia Study - Clean Dataset", layout="wide")
st.title("Clean & Filter Dataset")
st.caption(f"Source: `{config.DATASET_PATH}`")

if not config.DATASET_PATH.exists():
    st.warning("No dataset.csv found yet. Build it from the main page first ('Export ML dataset CSV').")
    st.stop()


@st.cache_data(ttl=60)
def load_dataset(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str, dtype=str)


df = load_dataset(str(config.DATASET_PATH), config.DATASET_PATH.stat().st_mtime)
corrections = cleaner.load_corrections(config.CORRECTIONS_PATH)

st.metric("Total rows", len(df))

# --- Missing image / Hb report -------------------------------------------------
report = cleaner.build_report(df, config.OUTPUT_DIR, corrections)

st.divider()
st.subheader("Missing Data")
col1, col2 = st.columns(2)
col1.metric("Rows with missing/unreadable image", int(report["img_missing_mask"].sum()))
col2.metric("Rows with missing/invalid hb%", int(report["hb_missing_mask"].sum()))
st.caption("These rows are dropped automatically -- there's nothing to map manually for them.")

# --- Manual mapping for uncommon category values -------------------------------
st.divider()
st.subheader("Category Value Mapping")
st.caption("Values that don't match the canonical list below. Assign each one to a category and save; "
           "the mapping is remembered for future dataset builds.")

for col, canonical in cleaner.CATEGORY_CANONICAL.items():
    unresolved = cleaner.unresolved_values(df[col], canonical, corrections.get(col, {}))

    st.markdown(f"**{col}** — canonical values: {', '.join(canonical)}")
    if not unresolved:
        st.success(f"All '{col}' values map cleanly to the canonical set.")
        continue

    st.warning(f"{len(unresolved)} unmapped value(s) found in '{col}' ({sum(c for _, c in unresolved)} rows).")

    pending = {}
    for value, count in unresolved:
        suggestion = cleaner.suggest_canonical(value, canonical)
        options = ["-- leave unmapped --"] + canonical
        default_index = options.index(suggestion) if suggestion else 0
        choice = st.selectbox(
            f"'{value}'  ({count} row{'s' if count != 1 else ''})",
            options=options,
            index=default_index,
            key=f"map_{col}_{value}",
        )
        if choice != "-- leave unmapped --":
            pending[value] = choice

    if st.button(f"Save '{col}' mappings", key=f"save_{col}"):
        corrections.setdefault(col, {}).update(pending)
        cleaner.save_corrections(config.CORRECTIONS_PATH, corrections)
        st.success(f"Saved {len(pending)} mapping(s) for '{col}'.")
        st.rerun()

# --- Numeric outlier report ------------------------------------------------------
st.divider()
st.subheader("Numeric Outlier Report")
st.caption("Values outside a physiologically plausible range. Flagged for review, not dropped unless you opt in below.")

outlier_masks = report["outlier_masks"]
outlier_cols = st.columns(len(outlier_masks))
for (col, mask), ui_col in zip(outlier_masks.items(), outlier_cols):
    lo, hi = cleaner.NUMERIC_RANGES[col]
    ui_col.metric(f"{col} outside [{lo}, {hi}]", int(mask.sum()))

any_outliers = any(mask.sum() > 0 for mask in outlier_masks.values())
if any_outliers:
    combined_outlier_mask = pd.Series(False, index=df.index)
    for mask in outlier_masks.values():
        combined_outlier_mask = combined_outlier_mask | mask
    with st.expander(f"View {int(combined_outlier_mask.sum())} flagged row(s)"):
        st.dataframe(df[combined_outlier_mask], use_container_width=True, hide_index=True)

exclude_outliers = st.checkbox("Exclude flagged outlier rows from the cleaned export", value=False)

# --- Demographic completeness report ----------------------------------------------
st.divider()
st.subheader("Demographic Completeness")
st.caption(
    "Rows with a valid image and hb% are kept in the export even if these fields are blank -- "
    "this is informational only, nothing here is dropped."
)

missing_field_masks = report["missing_field_masks"]
demo_cols = st.columns(len(missing_field_masks))
for (col, mask), ui_col in zip(missing_field_masks.items(), demo_cols):
    ui_col.metric(f"{col} missing", int(mask.sum()))

# --- Build cleaned dataset -------------------------------------------------------
st.divider()
st.subheader("Export Cleaned Dataset")

clean_df, needs_review_df, summary = cleaner.build_clean_dataset(
    df, config.OUTPUT_DIR, corrections, exclude_outliers=exclude_outliers
)

st.write(
    f"Usable rows: **{summary['usable']}** / {summary['total_rows']}  "
    f"(missing image: {summary['img_missing']}, missing hb%: {summary['hb_missing']}, "
    f"unresolved category: {summary['category_unresolved']}, "
    f"outliers excluded: {summary['outliers_excluded']}, "
    f"complete demographics: {summary['complete_demographics']}/{summary['usable']})"
)

st.caption("Usable rows by image modality:")
st.dataframe(
    pd.DataFrame(
        sorted(summary["modality_breakdown"].items()), columns=["image_type", "usable_rows"]
    ),
    use_container_width=True,
    hide_index=True,
)

if len(needs_review_df) > 0:
    st.warning(
        f"{len(needs_review_df)} row(s) are included in the export below, but with an unresolved "
        "category value left blank. Map them above and rebuild to fill the blanks in."
    )
    with st.expander("View rows needing review"):
        st.dataframe(needs_review_df, use_container_width=True, hide_index=True)

if st.button("Build cleaned dataset CSV", type="primary"):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(config.CLEAN_DATASET_PATH, index=False)
    cleaner.save_encoding_map(config.ENCODING_MAP_PATH)

    st.success(f"Saved {len(clean_df)} rows to {config.CLEAN_DATASET_PATH}")
    st.success(f"Saved label-encoding map to {config.ENCODING_MAP_PATH}")
    st.dataframe(clean_df.head(20), use_container_width=True, hide_index=True)

    dl_col1, dl_col2 = st.columns(2)
    dl_col1.download_button(
        "Download cleaned_dataset.csv",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )
    dl_col2.download_button(
        "Download encoding_map.json",
        data=json.dumps(cleaner.build_encoding_map(), indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="encoding_map.json",
        mime="application/json",
        use_container_width=True,
    )

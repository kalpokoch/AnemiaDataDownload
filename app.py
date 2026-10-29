import pandas as pd
import streamlit as st

import config
import downloader

st.set_page_config(page_title="Anemia Study - Image Downloader", layout="wide")

if not config.LOCAL_SHEET_PATH and not config.GOOGLE_SHEET_URL:
    st.error("GOOGLE_SHEET_URL is not set. Copy .env.example to .env and fill it in.")
    st.stop()


@st.cache_data(ttl=300)
def load_sheet_cached(sheet_url: str) -> pd.DataFrame:
    return downloader.load_sheet(sheet_url)


@st.cache_data(ttl=300)
def load_local_sheet_cached(path: str) -> pd.DataFrame:
    return downloader.load_local_sheet(path)


st.title("Anemia Study — Image Downloader")
st.caption("Pull module photos from the study's Google Sheet, crop them with the trained models, and export a training-ready dataset.")

with st.sidebar:
    st.header("Sheet Controls")
    if config.LOCAL_SHEET_PATH:
        st.caption(f"Using local sheet: {config.LOCAL_SHEET_PATH}")
        if st.button("Refresh sheet data", use_container_width=True):
            load_local_sheet_cached.clear()
            st.rerun()
    else:
        st.caption(f"Sheet ID: {downloader.sheet_url_to_csv_export(config.GOOGLE_SHEET_URL).split('/d/')[1].split('/')[0]}")
        if st.button("Refresh sheet data", use_container_width=True):
            load_sheet_cached.clear()
            st.rerun()

try:
    if config.LOCAL_SHEET_PATH:
        df = load_local_sheet_cached(config.LOCAL_SHEET_PATH)
    else:
        df = load_sheet_cached(config.GOOGLE_SHEET_URL)
except Exception as e:
    st.error(f"Could not load the sheet data.\n\n{e}")
    st.stop()

st.metric("Total rows in sheet", len(df))

st.divider()
st.subheader("Download Status by Module")
analytics_df = downloader.analyze_links(df, config.IMAGE_COLS, config.OUTPUT_DIR, config.NAME_COL)
st.dataframe(analytics_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Choose Modules to Download")

module_keys = list(config.IMAGE_COLS.keys())

def set_all_module_checks(is_selected: bool) -> None:
    for module_key in module_keys:
        st.session_state[f"chk_{module_key}"] = is_selected

col_a, col_b = st.columns(2)
col_a.button("Select all", use_container_width=True, on_click=set_all_module_checks, args=(True,))
col_b.button("Clear all", use_container_width=True, on_click=set_all_module_checks, args=(False,))

checkbox_cols = st.columns(3)
for i, module_key in enumerate(module_keys):
    widget_key = f"chk_{module_key}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = False
    checked = checkbox_cols[i % 3].checkbox(
        module_key, value=st.session_state[widget_key], key=widget_key
    )

selected_modules = [k for k in module_keys if st.session_state.get(f"chk_{k}", False)]

if st.button("Download ticked modules", disabled=not selected_modules, type="primary"):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if config.MANIFEST_PATH.exists():
        manifest_df = pd.read_csv(config.MANIFEST_PATH)
    else:
        manifest_df = df.copy()

    for module_key in selected_modules:
        column = config.IMAGE_COLS[module_key]

        with st.status(f"Downloading '{module_key}' ({column})...", expanded=True) as status_box:
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            log_box = st.container(height=150)

            def progress_cb(done, total, status):
                progress_bar.progress(done / total)
                status_placeholder.text(f"{done}/{total} processed - last status: {status}")
                if status not in ("ok", "exists"):
                    log_box.write(f"Row {done}: {status}")

            result_df = downloader.download_module(
                df, module_key, column, config.OUTPUT_DIR, config.NAME_COL,
                progress_cb, config.DOWNLOAD_WORKERS,
                model_path=config.YOLO_MODEL_PATHS.get(module_key),
                crop_conf=config.YOLO_CONF_THRESHOLD,
                crop_masked=module_key in config.YOLO_MASK_MODULES,
            )

            status_col = f"{column}__status"
            counts = result_df[status_col].apply(
                lambda s: "error" if str(s).startswith("error:") else s
            ).value_counts()
            status_box.update(label=f"'{module_key}' done: {counts.to_dict()}", state="complete")

            manifest_df[f"{column}__file"] = result_df[f"{column}__file"]
            manifest_df[f"{column}__status"] = result_df[status_col]

    manifest_df.to_csv(config.MANIFEST_PATH, index=False)
    st.success(f"Manifest updated: {config.MANIFEST_PATH}")

st.divider()
st.subheader("Crop Downloaded Images to Detected Region")
for crop_module_key, crop_model_path in config.YOLO_MODEL_PATHS.items():
    crop_column = config.IMAGE_COLS[crop_module_key]
    crop_target_dir = downloader.module_dir(config.OUTPUT_DIR, crop_column)
    existing_count = len(list(crop_target_dir.glob("*.jpg"))) if crop_target_dir.exists() else 0

    st.caption(f"**{crop_module_key}** — {existing_count} downloaded original(s)")
    overwrite_crops = st.checkbox("Overwrite existing crops", key=f"overwrite_{crop_module_key}")
    if st.button(f"Crop '{crop_module_key}' images", disabled=existing_count == 0, key=f"crop_{crop_module_key}"):
        crop_progress_bar = st.progress(0)
        crop_status_text = st.empty()

        def crop_progress(done, total, status):
            crop_progress_bar.progress(done / total)
            crop_status_text.text(f"{done}/{total} - {status}")

        crop_result = downloader.recrop_existing(
            config.OUTPUT_DIR, crop_column, crop_model_path, config.YOLO_CONF_THRESHOLD, crop_progress,
            masked=crop_module_key in config.YOLO_MASK_MODULES,
            overwrite=overwrite_crops,
        )
        st.success(f"'{crop_module_key}': {crop_result}")

st.divider()
st.subheader("Export ML Dataset CSV")
st.caption("One row per downloaded image, joined with demographic columns from the sheet.")

if st.button("Build dataset CSV", type="primary"):
    dataset_df = downloader.build_dataset(
        df, config.IMAGE_COLS, config.OUTPUT_DIR, config.NAME_COL,
        cropped_modules=set(config.YOLO_MODEL_PATHS),
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = config.OUTPUT_DIR / "dataset.csv"
    dataset_df.to_csv(dataset_path, index=False)

    st.success(f"Dataset written: {dataset_path} ({len(dataset_df)} rows)")
    st.dataframe(dataset_df.head(20), use_container_width=True, hide_index=True)
    st.download_button(
        "Download dataset.csv",
        data=dataset_df.to_csv(index=False).encode("utf-8"),
        file_name="dataset.csv",
        mime="text/csv",
    )

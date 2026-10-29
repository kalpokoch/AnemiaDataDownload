import operator
from functools import reduce

import pandas as pd
import streamlit as st

import cleaner
import config

st.set_page_config(page_title="Anemia Study - Usable Data by Modality", layout="wide")
st.title("Usable Data by Modality")
st.caption(f"Source: `{config.DATASET_PATH}`")

if not config.DATASET_PATH.exists():
    st.warning("No dataset.csv found yet. Build it from the main page first ('Export ML dataset CSV').")
    st.stop()

# dataset.csv is long-format: one row per (person, modality). "Usable" is a per-person,
# per-modality concept, so we pivot to one row per person with an img_path column per
# modality before computing masks.
DEMO_KEY_COLS = ["Name", "Age", "gender", "weight", "height", "race", "complexion", "hb%"]

MODALITY_LABELS = {
    "lower_eye_lid": "Lower Eyelid",
    "palm": "Palm",
    "fingertips": "Fingertips",
    "sclera": "Sclera",
    "tongue_front": "Tongue Front",
    "tongue_back": "Tongue Back",
}


@st.cache_data(ttl=60)
def load_dataset(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str, dtype=str)


@st.cache_data(ttl=60)
def build_person_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index=DEMO_KEY_COLS, columns="image_type", values="crop_path" if "crop_path" in df.columns else "img_path", aggfunc="first"
    ).reset_index()


df = load_dataset(str(config.DATASET_PATH), config.DATASET_PATH.stat().st_mtime)
person_df = build_person_table(df)

# Same hb% validity check used by the Clean Dataset page's "missing hb%" report.
hb_valid_mask = ~cleaner.missing_hb_mask(person_df)


def modality_usable_mask(modality_key: str) -> pd.Series:
    if modality_key not in person_df.columns:
        return pd.Series(False, index=person_df.index)
    img_frame = pd.DataFrame({"crop_path": person_df[modality_key]})
    img_valid_mask = ~cleaner.missing_image_mask(img_frame, config.OUTPUT_DIR)
    return img_valid_mask & hb_valid_mask


modality_masks = {key: modality_usable_mask(key) for key in MODALITY_LABELS}

# The pivot above (DEMO_KEY_COLS as index) silently drops any row with a blank hb% (or other
# key demo field), since pivot_table excludes NaN group keys. That makes person_df unusable for
# counting "missing hb%" -- those participants vanish before the check ever runs. Instead, dedupe
# the long-format dataset.csv by Name to get one row per participant (any of their modality rows
# carries the same hb%), then run the same validity check on that.
unique_participants_df = df.drop_duplicates(subset="Name")
total_participants = len(unique_participants_df)
missing_hb_count = int(cleaner.missing_hb_mask(unique_participants_df).sum())

total_col, missing_hb_col = st.columns(2)
total_col.metric("Total unique participants", total_participants)
missing_hb_col.metric("Missing valid hb%", f"{missing_hb_count} / {total_participants}")

st.divider()
st.subheader("Usable Rows per Modality")
st.caption(
    "A participant is 'usable' for a modality if that modality's image file exists on disk "
    "and hb% is present and numeric (same definition as the Missing Data report on the "
    "Clean Dataset page)."
)

metric_cols = st.columns(len(MODALITY_LABELS))
for ui_col, (key, label) in zip(metric_cols, MODALITY_LABELS.items()):
    ui_col.metric(label, int(modality_masks[key].sum()))

st.divider()
st.subheader("Modality Intersection")
st.caption(
    "Select modalities to see how many participants have usable data for ALL selected "
    "modalities simultaneously."
)

selected_labels = st.multiselect("Modalities", options=list(MODALITY_LABELS.values()))
label_to_key = {label: key for key, label in MODALITY_LABELS.items()}
selected_keys = [label_to_key[label] for label in selected_labels]

if not selected_keys:
    st.info("Select at least one modality above.")
else:
    combined_mask = reduce(operator.and_, (modality_masks[key] for key in selected_keys))
    st.metric(f"Usable for: {' + '.join(selected_labels)}", int(combined_mask.sum()))

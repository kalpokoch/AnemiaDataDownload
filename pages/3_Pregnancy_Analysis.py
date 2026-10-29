import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="Anemia Study - Pregnancy Analysis", layout="wide")
st.title("Pregnancy Analysis")
st.caption(f"Source: `{config.MANIFEST_PATH}` (Remarks column only exists on the raw sheet manifest, not dataset.csv)")

if not config.MANIFEST_PATH.exists():
    st.warning("No manifest found yet. Download at least one module from the main page first to generate it.")
    st.stop()

REMARKS_COL = "Remarks"
GENDER_COL = "Gender (M/F)"
HB_COL = "Hb%"


@st.cache_data(ttl=60)
def load_manifest(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str, dtype=str)


df = load_manifest(str(config.MANIFEST_PATH), config.MANIFEST_PATH.stat().st_mtime)

# Case-insensitive substring match on "preg" (not "pregnan") so typo variants of
# "Pregnant" (the sheet currently has "Pregant Women", missing the second "n") still match.
pregnant_mask = df[REMARKS_COL].astype(str).str.strip().str.lower().str.contains("preg", na=False)
hb_numeric = pd.to_numeric(df[HB_COL], errors="coerce")
hb_valid_mask = hb_numeric.notna()

st.metric("Total rows in sheet", len(df))

# --- Data quality flag: pregnant remark on a male row ---------------------------------
st.divider()
st.subheader("Data Quality Check")

bad_pregnant_mask = pregnant_mask & (df[GENDER_COL] == "M")
if bad_pregnant_mask.any():
    st.warning(
        f"{int(bad_pregnant_mask.sum())} row(s) are marked '{REMARKS_COL}' = pregnant but have "
        f"gender = M. These are likely data-entry errors, but are still counted as 'pregnant' "
        f"in the stats below."
    )
    with st.expander("View flagged rows"):
        st.dataframe(df[bad_pregnant_mask], use_container_width=True, hide_index=True)
else:
    st.success("No gender/pregnancy remark mismatches found.")

# --- Section A: overall pregnant vs non-pregnant counts -------------------------------
st.divider()
st.subheader("Pregnant vs Non-Pregnant — Overall")

col1, col2, col3 = st.columns(3)
col1.metric("Pregnant", int(pregnant_mask.sum()))
col2.metric("Non-pregnant", int((~pregnant_mask).sum()))
col3.metric("With valid Hb%", int(hb_valid_mask.sum()))

overall_table = pd.DataFrame({
    "Group": ["Pregnant", "Non-pregnant"],
    "Total rows": [int(pregnant_mask.sum()), int((~pregnant_mask).sum())],
    "With valid Hb%": [
        int((pregnant_mask & hb_valid_mask).sum()),
        int((~pregnant_mask & hb_valid_mask).sum()),
    ],
})
st.dataframe(overall_table, use_container_width=True, hide_index=True)

# --- Section B: pregnant vs non-pregnant at the Hb% tails -----------------------------
st.divider()
st.subheader("Pregnant vs Non-Pregnant at the Hb% Tails")
st.caption(
    "Rows with Hb% at or below the low threshold, or at or above the high threshold. "
    "Only rows with a valid numeric Hb% are considered."
)

thresh_col1, thresh_col2 = st.columns(2)
low_threshold = thresh_col1.number_input("Low Hb% threshold (severe/tail-low, inclusive)", value=7.0, step=0.5)
high_threshold = thresh_col2.number_input("High Hb% threshold (tail-high, inclusive)", value=15.0, step=0.5)

if low_threshold >= high_threshold:
    st.error("Low threshold must be less than high threshold.")
    st.stop()

low_tail_mask = hb_valid_mask & (hb_numeric <= low_threshold)
high_tail_mask = hb_valid_mask & (hb_numeric >= high_threshold)

tail_table = pd.DataFrame({
    "Tail": [f"Low (Hb% ≤ {low_threshold})", f"High (Hb% ≥ {high_threshold})"],
    "Pregnant": [
        int((pregnant_mask & low_tail_mask).sum()),
        int((pregnant_mask & high_tail_mask).sum()),
    ],
    "Non-pregnant": [
        int((~pregnant_mask & low_tail_mask).sum()),
        int((~pregnant_mask & high_tail_mask).sum()),
    ],
    "Total": [int(low_tail_mask.sum()), int(high_tail_mask.sum())],
})
st.dataframe(tail_table, use_container_width=True, hide_index=True)

with st.expander("View low-tail rows"):
    st.dataframe(df[low_tail_mask], use_container_width=True, hide_index=True)
with st.expander("View high-tail rows"):
    st.dataframe(df[high_tail_mask], use_container_width=True, hide_index=True)

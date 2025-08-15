import streamlit as st
import json
import zipfile
from pathlib import Path

st.set_page_config(page_title="JSON Viewer", layout="wide")

# --- Paths ---
ZIP_PATH = Path("data/data.zip")        # put your zip here
EXTRACT_DIR = Path("data/_extracted")   # unzip target

# --- Unzip on first run ---
def ensure_unzipped():
    if not ZIP_PATH.exists():
        st.error(f"Zip not found at {ZIP_PATH}. Put your data.zip in the data/ folder.")
        return False
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    # Only extract if empty
    if not any(EXTRACT_DIR.iterdir()):
        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.extractall(EXTRACT_DIR)
        except zipfile.BadZipFile:
            st.error("The file at data/data.zip is not a valid ZIP.")
            return False
    return True

@st.cache_data
def list_json_files():
    return sorted([p.name for p in EXTRACT_DIR.glob("*.json")])

@st.cache_data
def load_json(filename: str):
    with open(EXTRACT_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)

st.title("📂 Zipped JSON Viewer")

if ensure_unzipped():
    json_files = list_json_files()
    if not json_files:
        st.warning("No JSON files found inside the extracted ZIP.")
    else:
        selected = st.selectbox("Select a JSON file:", json_files)
        data = load_json(selected)

        left, right = st.columns([1, 2], gap="large")
        with left:
            st.markdown("**File**")
            st.code(selected)
            st.markdown("**Type**")
            st.code(type(data).__name__)
            if hasattr(data, "__len__"):
                st.markdown("**Length**")
                st.code(len(data))

        with right:
            st.markdown("**Preview**")
            if isinstance(data, dict):
                # Show first 10 keys
                keys = list(data.keys())[:10]
                st.json({k: data[k] for k in keys})
            elif isinstance(data, list):
                st.json(data[:5])
            else:
                st.write(data)

        st.divider()
        st.caption("Tip: Keep the repo light. For very large data, host externally and fetch at runtime.")

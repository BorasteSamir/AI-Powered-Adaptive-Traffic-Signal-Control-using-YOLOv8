# model_loader.py — YOLOv8 model loading, cached for the Streamlit session.

import os
import streamlit as st
from typing import Optional

try:
    from ultralytics import YOLO
except ImportError:
    st.error("❌ ultralytics not installed. Run: pip install ultralytics")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Optional[YOLO]:
    """
    Load YOLOv8 model from disk.
    Cached so it is loaded only once per Streamlit session.
    Returns None if the file does not exist or loading fails.
    """
    if not os.path.exists(model_path):
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

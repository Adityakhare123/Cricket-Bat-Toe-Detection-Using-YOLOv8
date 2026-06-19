from pathlib import Path

import requests
import streamlit as st


def download_model_if_missing(model_path: str, model_url: str) -> str:
    """
    Download YOLO model if it is missing locally.

    Local:
    - Uses models/cricket_bat_toe_best.pt if present.

    Streamlit Cloud:
    - Downloads model from MODEL_URL secret.
    """

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    if model_file.exists():
        return str(model_file)

    if not model_url:
        st.error(
            "Model file is missing and MODEL_URL is not configured.\n\n"
            "For Streamlit Cloud deployment, add MODEL_URL in app secrets."
        )
        st.stop()

    try:
        with st.spinner("Downloading YOLO model..."):
            response = requests.get(model_url, stream=True, timeout=180)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0
            progress_bar = st.progress(0)

            with open(model_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)

                        if total_size > 0:
                            progress_bar.progress(
                                min(downloaded_size / total_size, 1.0)
                            )

            progress_bar.empty()

        return str(model_file)

    except Exception as error:
        st.error(f"Failed to download model: {error}")
        st.stop()
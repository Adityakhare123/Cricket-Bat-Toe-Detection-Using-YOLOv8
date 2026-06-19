import os
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

from utils.detector import CricketBatToeDetector
from utils.video_utils import process_video
from utils.model_loader import download_model_if_missing


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cricket Bat Toe Detection",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS + MODEL DOWNLOAD
# ============================================================

LOCAL_MODEL_PATH = "models/cricket_bat_toe_best.pt"
MODEL_URL = os.getenv("MODEL_URL", "")

MODEL_PATH = download_model_if_missing(
    model_path=LOCAL_MODEL_PATH,
    model_url=MODEL_URL
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.16), transparent 35%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.16), transparent 35%),
                linear-gradient(135deg, #020617 0%, #07111f 45%, #020617 100%);
            color: #ffffff;
        }

        section[data-testid="stSidebar"] {
            background: rgba(2, 6, 23, 0.94);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        .hero-card {
            padding: 38px;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.92)),
                linear-gradient(135deg, rgba(34, 197, 94, 0.35), rgba(59, 130, 246, 0.28));
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
            margin-bottom: 28px;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 14px;
            background: linear-gradient(90deg, #ffffff, #86efac, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            font-size: 1.08rem;
            color: #cbd5e1;
            max-width: 820px;
            line-height: 1.7;
        }

        .badge-row {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 24px;
        }

        .premium-badge {
            padding: 9px 14px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.25);
            color: #dbeafe;
            font-size: 0.88rem;
            font-weight: 600;
        }

        .glass-card {
            padding: 24px;
            border-radius: 24px;
            background: rgba(15, 23, 42, 0.76);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.28);
            margin-bottom: 20px;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 800;
            margin-bottom: 8px;
            color: #ffffff;
        }

        .section-subtitle {
            font-size: 0.96rem;
            color: #94a3b8;
            margin-bottom: 20px;
        }

        .metric-card {
            padding: 20px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(2, 6, 23, 0.85), rgba(15, 23, 42, 0.85));
            border: 1px solid rgba(148, 163, 184, 0.18);
            text-align: center;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #86efac;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #cbd5e1;
            margin-top: 4px;
        }

        .legend-box {
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(2, 6, 23, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            margin-top: 14px;
        }

        .blue-dot, .green-dot, .red-dot {
            height: 12px;
            width: 12px;
            border-radius: 999px;
            display: inline-block;
            margin-right: 8px;
        }

        .blue-dot {
            background: #3b82f6;
        }

        .green-dot {
            background: #22c55e;
        }

        .red-dot {
            background: #ef4444;
        }

        div[data-testid="stFileUploader"] {
            padding: 18px;
            border-radius: 22px;
            background: rgba(2, 6, 23, 0.7);
            border: 1px dashed rgba(148, 163, 184, 0.35);
        }

        .stButton > button {
            width: 100%;
            border-radius: 16px;
            padding: 0.85rem 1.2rem;
            font-weight: 800;
            border: none;
            background: linear-gradient(90deg, #22c55e, #3b82f6);
            color: white;
            box-shadow: 0 12px 32px rgba(34, 197, 94, 0.22);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 42px rgba(59, 130, 246, 0.32);
        }

        .stDownloadButton > button {
            border-radius: 14px;
            padding: 0.75rem 1rem;
            font-weight: 700;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.35);
            color: #e5e7eb;
        }

        div[data-testid="stTabs"] button {
            font-weight: 700;
            color: #cbd5e1;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        .footer {
            text-align: center;
            color: #64748b;
            margin-top: 32px;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_detector():
    return CricketBatToeDetector(model_path=MODEL_PATH)


# ============================================================
# CLASS MAPPING OPTIONS
# ============================================================

CLASS_MAPPING_OPTIONS = {
    "1 = Bat, 2/3 = Toe": {1: "Bat", 2: "Toe", 3: "Toe"},
    "0 = Bat, 1 = Toe": {0: "Bat", 1: "Toe"},
    "0 = Toe, 1 = Bat": {0: "Toe", 1: "Bat"},
    "1 = Bat, 2 = Toe": {1: "Bat", 2: "Toe"},
    "1 = Toe, 2 = Bat": {1: "Toe", 2: "Bat"},
    "2 = Bat, 1 = Toe": {2: "Bat", 1: "Toe"},
    "2 = Toe, 1 = Bat": {2: "Toe", 1: "Bat"},
    "0 = Bat, 2 = Toe": {0: "Bat", 2: "Toe"},
    "0 = Toe, 2 = Bat": {0: "Toe", 2: "Bat"},
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🏏 Control Panel")
st.sidebar.caption("Tune detection settings and class mapping.")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05
)

iou = st.sidebar.slider(
    "IOU Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.45,
    step=0.05
)

st.sidebar.markdown("---")

mapping_option = st.sidebar.selectbox(
    "Class Mapping",
    list(CLASS_MAPPING_OPTIONS.keys()),
    index=0
)

selected_class_names = CLASS_MAPPING_OPTIONS[mapping_option]

st.sidebar.markdown(
    f"""
    <div class="legend-box">
        <b>Selected Mapping</b><br><br>
        <code>{selected_class_names}</code>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="legend-box">
        <b>Detection Colors</b><br><br>
        <span class="blue-dot"></span> Bat<br>
        <span class="green-dot"></span> Toe<br>
        <span class="red-dot"></span> Unknown
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Cricket Bat Toe Detection</div>
        <div class="hero-subtitle">
            A computer vision interface powered by YOLOv8 and OpenCV for detecting cricket bats
            and localizing the toe region from images and videos.
        </div>
        <div class="badge-row">
            <div class="premium-badge">YOLOv8 Object Detection</div>
            <div class="premium-badge">OpenCV Video Processing</div>
            <div class="premium-badge">Image + Video Inference</div>
            <div class="premium-badge">Bat / Toe Localization</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DETECTOR
# ============================================================

detector = load_detector()


# ============================================================
# MODEL STATUS CARDS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">2</div>
            <div class="metric-label">Core Objects</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">YOLO</div>
            <div class="metric-label">Detection Engine</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">CV</div>
            <div class="metric-label">Video Pipeline</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">Live</div>
            <div class="metric-label">Inference Ready</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# MODEL INFO
# ============================================================

with st.expander("Advanced Model Information"):
    try:
        st.write("Model path:", MODEL_PATH)
        st.write("YOLO model class names:", detector.model.names)
        st.write("Selected UI mapping:", selected_class_names)
    except Exception:
        st.warning("Could not read model information.")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🖼️ Image Detection",
        "🎥 Video Detection",
        "📌 Project Details"
    ]
)


# ============================================================
# IMAGE DETECTION TAB
# ============================================================

with tab1:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Image Detection</div>
            <div class="section-subtitle">
                Upload a cricket bat image and run object detection to identify Bat and Toe regions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### Detection Output")
            output_placeholder = st.empty()

        run_image_detection = st.button("Run Image Detection")

        if run_image_detection:
            with st.spinner("Running YOLOv8 inference on image..."):
                result_image, detections = detector.predict_image(
                    image=image,
                    conf=confidence,
                    iou=iou,
                    class_names=selected_class_names
                )

            output_placeholder.image(result_image, use_container_width=True)

            st.markdown("### Detection Summary")

            if len(detections) == 0:
                st.warning("No objects detected. Try lowering the confidence threshold.")
            else:
                bat_count = sum(1 for d in detections if d["class_name"] == "Bat")
                toe_count = sum(1 for d in detections if d["class_name"] == "Toe")
                unknown_count = sum(1 for d in detections if "Unknown" in d["class_name"])

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric("Bat", bat_count)

                with c2:
                    st.metric("Toe", toe_count)

                with c3:
                    st.metric("Unknown", unknown_count)

                st.dataframe(detections, use_container_width=True)

            output_path = OUTPUT_DIR / f"image_detection_output_{int(time.time())}.jpg"

            cv2.imwrite(
                str(output_path),
                cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            )

            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Output Image",
                    data=file,
                    file_name="cricket_bat_toe_output.jpg",
                    mime="image/jpeg"
                )


# ============================================================
# VIDEO DETECTION TAB
# ============================================================

with tab2:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Video Detection</div>
            <div class="section-subtitle">
                Upload a cricket bat video and generate an annotated output video with Bat and Toe labels.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:
        st.markdown("### Original Video")
        st.video(uploaded_video)

        st.info(
            "Current recommended mapping is `1 = Bat, 2/3 = Toe`. "
            "If labels look wrong, change mapping from the sidebar."
        )

        run_video_detection = st.button("Run Video Detection")

        if run_video_detection:
            suffix = Path(uploaded_video.name).suffix

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
                temp_video.write(uploaded_video.read())
                input_video_path = temp_video.name

            output_video_path = OUTPUT_DIR / f"video_detection_output_{int(time.time())}.mp4"

            with st.spinner("Processing video frame by frame..."):
                process_video(
                    model_path=MODEL_PATH,
                    input_video_path=input_video_path,
                    output_video_path=str(output_video_path),
                    conf=confidence,
                    iou=iou,
                    class_names=selected_class_names
                )

            st.success("Video processing completed.")

            st.markdown("### Detection Output")
            st.video(str(output_video_path))

            with open(output_video_path, "rb") as file:
                st.download_button(
                    label="Download Output Video",
                    data=file,
                    file_name="cricket_bat_toe_video_output.mp4",
                    mime="video/mp4"
                )

            try:
                os.remove(input_video_path)
            except Exception:
                pass


# ============================================================
# PROJECT DETAILS TAB
# ============================================================

with tab3:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Project Overview</div>
            <div class="section-subtitle">
                A custom computer vision system for cricket equipment analysis.
            </div>

            <h4>Objective</h4>
            <p>
                The objective is to detect cricket bats and precisely localize the toe region,
                enabling future use cases like toe guard detection, wear analysis, and bat quality inspection.
            </p>

            <h4>Technology Stack</h4>
            <p>
                Python, YOLOv8, OpenCV, Streamlit, NumPy, Pillow
            </p>

            <h4>Current Capability</h4>
            <p>
                The system supports image and video inference, custom class mapping,
                colored bounding boxes, and downloadable prediction outputs.
            </p>

            <h4>Future Enhancements</h4>
            <ul>
                <li>Toe guard detection</li>
                <li>Crack and damage classification</li>
                <li>Real-time webcam detection</li>
                <li>Model retraining with cleaner class labels</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with YOLOv8, OpenCV and Streamlit · Cricket Bat Toe Detection
    </div>
    """,
    unsafe_allow_html=True
)
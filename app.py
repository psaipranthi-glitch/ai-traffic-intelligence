import streamlit as st
import cv2
import re
import os
import pandas as pd
from ultralytics import YOLO
import easyocr


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# GLOBAL THEME
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ========================================================
       ROOT
       ======================================================== */

    :root {
        --bg: #070B0F;
        --panel: #0D1319;
        --panel2: #111920;
        --border: #202B35;
        --border2: #2B3945;

        --text: #E8EEF3;
        --muted: #7F8D99;
        --dim: #53616D;

        --cyan: #00E6C3;
        --cyan-soft: rgba(0,230,195,0.08);

        --red: #FF5555;
        --orange: #FFB454;
        --blue: #4DA3FF;
    }


    /* ========================================================
       APP
       ======================================================== */

    html,
    body,
    [class*="css"] {
        font-family: "IBM Plex Sans", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 8% 0%,
                rgba(0,230,195,0.065),
                transparent 27%
            ),
            radial-gradient(
                circle at 92% 8%,
                rgba(77,163,255,0.04),
                transparent 25%
            ),
            var(--bg);

        color: var(--text);
    }


    /* ========================================================
       HIDE STREAMLIT DEFAULT UI
       ======================================================== */

    header[data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .stDeployButton {
        display: none;
    }


    /* ========================================================
       MAIN CONTAINER
       ======================================================== */

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        position: relative;

        width: 100%;
        box-sizing: border-box;

        padding: 25px 28px;
        margin-bottom: 26px;

        background:
            linear-gradient(
                135deg,
                rgba(17,25,32,0.98),
                rgba(9,14,19,0.98)
            );

        border: 1px solid var(--border);
        border-radius: 12px;

        overflow: hidden;

        box-shadow:
            0 20px 60px rgba(0,0,0,0.28);
    }

    .hero::before {
        content: "";

        position: absolute;

        top: 0;
        left: 0;

        width: 100%;
        height: 2px;

        background:
            linear-gradient(
                90deg,
                transparent,
                var(--cyan),
                transparent
            );

        opacity: 0.8;
    }

    .hero-grid {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 25px;
    }

    .hero-title {
        font-family: "JetBrains Mono", monospace;

        font-size: 25px;
        line-height: 1.2;

        font-weight: 700;

        letter-spacing: 0.5px;

        color: var(--text);
    }

    .hero-subtitle {
        margin-top: 9px;

        font-size: 13px;

        color: var(--muted);

        letter-spacing: 0.25px;
    }

    .hero-tech {
        display: flex;

        flex-wrap: wrap;

        gap: 7px;

        margin-top: 14px;
    }

    .tech {
        display: inline-block;

        padding: 5px 9px;

        border: 1px solid var(--border2);

        border-radius: 5px;

        background: rgba(255,255,255,0.018);

        color: #9DAAB5;

        font-family: "JetBrains Mono", monospace;

        font-size: 9px;

        letter-spacing: 0.7px;
    }

    .system-status {
        display: flex;

        align-items: center;

        gap: 9px;

        flex-shrink: 0;

        padding: 8px 13px;

        border:
            1px solid rgba(0,230,195,0.28);

        border-radius: 20px;

        background:
            rgba(0,230,195,0.055);

        color: var(--cyan);

        font-family: "JetBrains Mono", monospace;

        font-size: 10px;

        font-weight: 600;

        letter-spacing: 1px;

        white-space: nowrap;
    }

    .status-dot {
        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: var(--cyan);

        box-shadow:
            0 0 0 3px rgba(0,230,195,0.08),
            0 0 12px rgba(0,230,195,0.75);
    }


    /* ========================================================
       SECTION HEADER
       ======================================================== */

    .section-header {
        display: flex;

        align-items: center;

        gap: 10px;

        margin-top: 28px;

        margin-bottom: 13px;
    }

    .section-line {
        width: 3px;
        height: 18px;

        flex-shrink: 0;

        border-radius: 2px;

        background: var(--cyan);

        box-shadow:
            0 0 10px rgba(0,230,195,0.35);
    }

    .section-title {
        font-family: "JetBrains Mono", monospace;

        font-size: 11px;

        font-weight: 600;

        letter-spacing: 1.7px;

        text-transform: uppercase;

        color: #8D9AA6;
    }

    .section-meta {
        margin-left: auto;

        font-family: "JetBrains Mono", monospace;

        font-size: 9px;

        letter-spacing: 0.8px;

        color: var(--dim);

        text-transform: uppercase;
    }


    /* ========================================================
       UPLOAD PANEL
       ======================================================== */

    .upload-panel {
        width: 100%;
        box-sizing: border-box;

        padding: 18px;

        background:
            rgba(13,19,25,0.88);

        border:
            1px solid var(--border);

        border-radius: 9px;

        margin-bottom: 10px;
    }

    .upload-title {
        font-family: "JetBrains Mono", monospace;

        font-size: 11px;

        font-weight: 600;

        letter-spacing: 1px;

        color: var(--text);
    }

    .upload-description {
        margin-top: 6px;

        font-size: 11px;

        line-height: 1.6;

        color: var(--muted);
    }

    .upload-meta {
        margin-top: 10px;

        font-family: "JetBrains Mono", monospace;

        font-size: 9px;

        letter-spacing: 0.5px;

        color: var(--dim);
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        width: 100%;

        box-sizing: border-box;

        margin-top: 0;

        background:
            rgba(13,19,25,0.75);

        border: 1px dashed #33414D;

        border-radius: 9px;

        padding: 14px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color:
            rgba(0,230,195,0.45);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;

        border: none !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--muted) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: var(--muted) !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background:
            rgba(0,230,195,0.07) !important;

        border:
            1px solid rgba(0,230,195,0.35) !important;

        color: var(--cyan) !important;

        border-radius: 6px !important;
    }


    /* ========================================================
       EMPTY STATE
       ======================================================== */

    .empty-state {
        width: 100%;
        box-sizing: border-box;

        padding: 38px 20px;

        margin-top: 10px;

        text-align: center;

        background:
            rgba(13,19,25,0.52);

        border:
            1px dashed #293640;

        border-radius: 9px;
    }

    .empty-icon {
        font-size: 28px;

        margin-bottom: 10px;

        opacity: 0.55;
    }

    .empty-title {
        font-family: "JetBrains Mono", monospace;

        font-size: 10px;

        letter-spacing: 1px;

        color: #667581;
    }

    .empty-description {
        margin-top: 8px;

        font-size: 11px;

        color: var(--dim);
    }


    /* ========================================================
       STATUS BAR
       ======================================================== */

    .status-bar {
        display: flex;

        flex-wrap: wrap;

        align-items: center;

        gap: 9px;

        width: 100%;

        box-sizing: border-box;

        margin-top: 10px;

        padding: 9px 12px;

        background:
            rgba(13,19,25,0.8);

        border:
            1px solid var(--border);

        border-radius: 6px;

        font-family: "JetBrains Mono", monospace;

        font-size: 9px;

        letter-spacing: 0.5px;

        color: #697883;
    }

    .status-active {
        color: var(--cyan);
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton {
        margin-top: 10px;
    }

    .stButton > button {
        width: 100%;

        min-height: 44px;

        border:
            1px solid rgba(0,230,195,0.55) !important;

        border-radius: 7px !important;

        background:
            linear-gradient(
                135deg,
                rgba(0,230,195,0.13),
                rgba(0,230,195,0.045)
            ) !important;

        color: var(--cyan) !important;

        font-family: "JetBrains Mono", monospace !important;

        font-size: 11px !important;

        font-weight: 600 !important;

        letter-spacing: 0.7px !important;
    }

    .stButton > button:hover {
        border-color:
            var(--cyan) !important;

        background:
            rgba(0,230,195,0.14) !important;

        box-shadow:
            0 0 20px rgba(0,230,195,0.08);
    }


    /* ========================================================
       LIVE FEED
       ======================================================== */

    .live-badge {
        display: inline-flex;

        align-items: center;

        gap: 7px;

        padding: 5px 9px;

        margin-bottom: 8px;

        border-radius: 4px;

        background:
            rgba(7,11,15,0.9);

        border:
            1px solid rgba(255,70,70,0.3);

        color: #FF7777;

        font-family: "JetBrains Mono", monospace;

        font-size: 9px;

        font-weight: 600;

        letter-spacing: 1px;
    }

    .live-dot {
        width: 6px;
        height: 6px;

        border-radius: 50%;

        background: var(--red);

        box-shadow:
            0 0 9px rgba(255,70,70,0.7);
    }

    .feed-frame {
        width: 100%;

        padding: 7px;

        box-sizing: border-box;

        background: #090E13;

        border:
            1px solid var(--border);

        border-radius: 10px;
    }

    [data-testid="stImage"] {
        width: 100%;

        padding: 0 !important;

        background: #080C10;

        border: none !important;

        border-radius: 6px;

        overflow: hidden;
    }

    [data-testid="stImage"] img {
        border-radius: 5px;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {
        position: relative;

        min-height: 102px;

        box-sizing: border-box;

        padding: 17px 18px;

        background:
            linear-gradient(
                145deg,
                rgba(17,25,32,0.96),
                rgba(11,16,21,0.96)
            );

        border:
            1px solid var(--border);

        border-radius: 9px;

        overflow: hidden;
    }

    .metric-card:hover {
        border-color:
            rgba(0,230,195,0.32);
    }

    .metric-card::after {
        content: "";

        position: absolute;

        right: -25px;

        bottom: -30px;

        width: 90px;
        height: 90px;

        border-radius: 50%;

        background:
            rgba(0,230,195,0.035);
    }

    .metric-label {
        font-family: "JetBrains Mono", monospace;

        font-size: 9px;

        letter-spacing: 1.1px;

        text-transform: uppercase;

        color: #687782;
    }

    .metric-value {
        margin-top: 10px;

        font-family: "JetBrains Mono", monospace;

        font-size: 28px;

        line-height: 1;

        font-weight: 600;

        color: var(--text);
    }

    .metric-value.accent {
        color: var(--cyan);

        text-shadow:
            0 0 18px rgba(0,230,195,0.15);
    }

    .metric-icon {
        position: absolute;

        right: 15px;

        top: 14px;

        font-size: 16px;

        opacity: 0.5;
    }


    /* ========================================================
       TABLE
       ======================================================== */

    [data-testid="stDataFrame"] {
        width: 100%;

        border:
            1px solid var(--border);

        border-radius: 8px;

        overflow: hidden;
    }


    /* ========================================================
       PROGRESS
       ======================================================== */

    [data-testid="stProgress"] {
        margin-top: 10px;
    }

    [data-testid="stProgressBar"] {
        background: #151D24 !important;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    .stAlert {
        border-radius: 7px !important;

        border:
            1px solid var(--border) !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .app-footer {
        margin-top: 45px;

        padding-top: 18px;

        border-top:
            1px solid var(--border);

        text-align: center;

        font-family: "JetBrains Mono", monospace;

        font-size: 9px;

        letter-spacing: 1px;

        color: #46535E;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 800px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero-grid {
            flex-direction: column;
        }

        .system-status {
            align-self: flex-start;
        }

        .section-meta {
            display: none;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-grid">

            <div>

                <div class="hero-title">
                    🚦 AI TRAFFIC INTELLIGENCE
                </div>

                <div class="hero-subtitle">
                    Real-time vehicle detection, tracking
                    and license plate recognition
                </div>

                <div class="hero-tech">

                    <span class="tech">YOLO11</span>

                    <span class="tech">BYTE TRACK</span>

                    <span class="tech">PLATE DETECTION</span>

                    <span class="tech">EASYOCR</span>

                </div>

            </div>

            <div class="system-status">

                <span class="status-dot"></span>

                SYSTEM ONLINE

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section_header(title, meta=""):

    meta_html = ""

    if meta:
        meta_html = (
            f'<div class="section-meta">{meta}</div>'
        )

    st.markdown(
        f"""
        <div class="section-header">

            <div class="section-line"></div>

            <div class="section-title">
                {title}
            </div>

            {meta_html}

        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card(label, value, icon="", accent=False):

    value_class = (
        "metric-value accent"
        if accent
        else
        "metric-value"
    )

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-icon">
                {icon}
            </div>

            <div class="metric-label">
                {label}
            </div>

            <div class="{value_class}">
                {value}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def clean_text(text):

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


def create_plate_dataframe(plate_data):

    rows = []

    for vehicle_id, data in plate_data.items():

        rows.append(
            {
                "Vehicle ID": f"#{vehicle_id}",
                "License Plate": data["text"],
                "Confidence": round(
                    data["confidence"],
                    2
                )
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Vehicle ID",
                "License Plate",
                "Confidence"
            ]
        )

    df = pd.DataFrame(rows)

    df["_id"] = (
        df["Vehicle ID"]
        .str.replace(
            "#",
            "",
            regex=False
        )
        .astype(int)
    )

    df = (
        df
        .sort_values("_id")
        .drop(columns="_id")
    )

    return df


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():

    vehicle_model = YOLO(
        "yolo11n.pt"
    )

    plate_model = YOLO(
        "models/best.pt"
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False
    )

    return (
        vehicle_model,
        plate_model,
        reader
    )


# ============================================================
# INPUT FEED
# ============================================================

section_header(
    "Input Feed",
    "VIDEO ANALYSIS PIPELINE"
)

st.markdown(
    """
    <div class="upload-panel">

        <div class="upload-title">
            TRAFFIC VIDEO SOURCE
        </div>

        <div class="upload-description">
            Upload MP4, AVI, MOV or MKV footage for
            AI-powered traffic analysis.
            Maximum file size: 200 MB.
        </div>

        <div class="upload-meta">
            200 MB PER FILE &nbsp;•&nbsp;
            MP4 &nbsp;·&nbsp; AVI &nbsp;·&nbsp; MOV &nbsp;·&nbsp; MKV
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded = st.file_uploader(
    "Traffic video source",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ],
    label_visibility="collapsed"
)


# ============================================================
# NO FILE
# ============================================================

if uploaded is None:

    st.markdown(
        """
        <div class="empty-state">

            <div class="empty-icon">
                🎥
            </div>

            <div class="empty-title">
                AWAITING VIDEO INPUT
            </div>

            <div class="empty-description">
                Upload traffic footage to initialize
                the AI detection pipeline
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# FILE SIZE
# ============================================================

file_size_mb = uploaded.size / (
    1024 * 1024
)


if file_size_mb > 200:

    st.error(
        "Video exceeds the 200 MB limit."
    )

    st.stop()


# ============================================================
# FILE READY
# ============================================================

st.markdown(
    f"""
    <div class="status-bar">

        <span class="status-active">
            ● FILE READY
        </span>

        <span>
            {uploaded.name}
        </span>

        <span>·</span>

        <span>
            {file_size_mb:.1f} MB
        </span>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# START
# ============================================================

start = st.button(
    "🚀  START AI TRAFFIC ANALYSIS",
    type="primary",
    use_container_width=True
)


if not start:
    st.stop()


# ============================================================
# SAVE VIDEO
# ============================================================

video_path = "uploaded_traffic.mp4"

with open(
    video_path,
    "wb"
) as file:

    file.write(
        uploaded.getbuffer()
    )


# ============================================================
# AI ENGINE
# ============================================================

section_header(
    "AI Engine",
    "INITIALIZATION"
)

with st.spinner(
    "Loading YOLO11, ByteTrack, plate detector and EasyOCR..."
):

    (
        vehicle_model,
        plate_model,
        reader
    ) = load_models()


st.success(
    "AI inference engine initialized successfully."
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    video_path
)


if not cap.isOpened():

    st.error(
        "Could not open the uploaded video."
    )

    st.stop()


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


fps = cap.get(
    cv2.CAP_PROP_FPS
)


if fps <= 0:
    fps = 30


# ============================================================
# PROCESSING STATE
# ============================================================

frame_number = 0

processed_frames = 0

all_ids = set()

plate_data = {}

last_ocr = {}


FRAME_SKIP = 3

OCR_INTERVAL = 45


# ============================================================
# LIVE FEED
# ============================================================

section_header(
    "Live Feed",
    "REAL-TIME INFERENCE"
)

st.markdown(
    """
    <div class="live-badge">

        <span class="live-dot"></span>

        LIVE INFERENCE

    </div>
    """,
    unsafe_allow_html=True
)


video_display = st.empty()

progress = st.progress(0)

status_display = st.empty()


# ============================================================
# LIVE STATISTICS
# ============================================================

section_header(
    "Live Traffic Statistics",
    "CURRENT FRAME"
)

c1, c2, c3, c4, c5 = st.columns(5)

tracked_box = c1.empty()
cars_box = c2.empty()
bikes_box = c3.empty()
buses_box = c4.empty()
trucks_box = c5.empty()


# ============================================================
# LICENSE PLATES
# ============================================================

section_header(
    "License Plates",
    "OCR DETECTIONS"
)

plate_display = st.empty()


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    if frame_number % FRAME_SKIP != 0:
        continue

    processed_frames += 1


    # ========================================================
    # YOLO TRACKING
    # ========================================================

    results = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.40,
        iou=0.5,
        verbose=False
    )

    result = results[0]


    current_ids = set()

    cars = 0
    motorcycles = 0
    buses = 0
    trucks = 0


    # ========================================================
    # DETECTIONS
    # ========================================================

    if result.boxes is not None:

        boxes = result.boxes


        if boxes.id is not None:

            ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

            classes = (
                boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            coordinates = (
                boxes.xyxy
                .int()
                .cpu()
                .tolist()
            )


            for vehicle_id, cls, box in zip(
                ids,
                classes,
                coordinates
            ):

                current_ids.add(
                    vehicle_id
                )

                all_ids.add(
                    vehicle_id
                )


                # =================================================
                # VEHICLE CLASS
                # =================================================

                if cls == 2:

                    vehicle_type = "Car"

                    cars += 1

                elif cls == 3:

                    vehicle_type = "Motorcycle"

                    motorcycles += 1

                elif cls == 5:

                    vehicle_type = "Bus"

                    buses += 1

                elif cls == 7:

                    vehicle_type = "Truck"

                    trucks += 1

                else:

                    continue


                # =================================================
                # BOUNDING BOX
                # =================================================

                x1, y1, x2, y2 = box

                x1 = max(
                    0,
                    x1
                )

                y1 = max(
                    0,
                    y1
                )

                x2 = min(
                    frame.shape[1],
                    x2
                )

                y2 = min(
                    frame.shape[0],
                    y2
                )


                if (
                    x2 <= x1
                    or
                    y2 <= y1
                ):
                    continue


                crop = frame[
                    y1:y2,
                    x1:x2
                ]


                # =================================================
                # OCR
                # =================================================

                should_ocr = (
                    vehicle_id not in last_ocr
                    or
                    frame_number -
                    last_ocr[vehicle_id]
                    >= OCR_INTERVAL
                )


                if (
                    should_ocr
                    and crop.size > 0
                ):

                    last_ocr[
                        vehicle_id
                    ] = frame_number


                    try:

                        plate_results = plate_model(
                            crop,
                            conf=0.50,
                            verbose=False
                        )


                        for plate_result in plate_results:

                            if (
                                plate_result.boxes
                                is None
                            ):
                                continue


                            plate_boxes = (
                                plate_result
                                .boxes
                                .xyxy
                                .int()
                                .cpu()
                                .tolist()
                            )


                            for pbox in plate_boxes:

                                px1, py1, px2, py2 = pbox


                                px1 = max(
                                    0,
                                    px1
                                )

                                py1 = max(
                                    0,
                                    py1
                                )

                                px2 = min(
                                    crop.shape[1],
                                    px2
                                )

                                py2 = min(
                                    crop.shape[0],
                                    py2
                                )


                                if (
                                    px2 <= px1
                                    or
                                    py2 <= py1
                                ):
                                    continue


                                plate_crop = crop[
                                    py1:py2,
                                    px1:px2
                                ]


                                if plate_crop.size == 0:
                                    continue


                                try:

                                    ocr_results = reader.readtext(
                                        plate_crop,
                                        detail=1
                                    )

                                except Exception:

                                    ocr_results = []


                                best_text = ""

                                best_conf = 0.0


                                for item in ocr_results:

                                    if len(item) < 3:
                                        continue


                                    text = clean_text(
                                        item[1]
                                    )


                                    confidence = float(
                                        item[2]
                                    )


                                    if (
                                        len(text) >= 3
                                        and
                                        confidence > best_conf
                                    ):

                                        best_text = text

                                        best_conf = confidence


                                if best_text:

                                    previous = plate_data.get(
                                        vehicle_id
                                    )


                                    if (
                                        previous is None
                                        or
                                        best_conf >
                                        previous["confidence"]
                                    ):

                                        plate_data[
                                            vehicle_id
                                        ] = {
                                            "text": best_text,
                                            "confidence": best_conf
                                        }


                    except Exception:
                        pass


                # =================================================
                # DRAW VEHICLE BOX
                # =================================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (195, 230, 0),
                    2
                )


                # =================================================
                # VEHICLE LABEL
                # =================================================

                cv2.putText(
                    frame,
                    f"{vehicle_type} ID:{vehicle_id}",
                    (
                        x1,
                        max(
                            25,
                            y1 - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (195, 230, 0),
                    2
                )


                # =================================================
                # PLATE LABEL
                # =================================================

                if vehicle_id in plate_data:

                    plate_text = plate_data[
                        vehicle_id
                    ]["text"]


                    cv2.putText(
                        frame,
                        f"Plate: {plate_text}",
                        (
                            x1,
                            min(
                                frame.shape[0] - 10,
                                y2 + 22
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (32, 176, 255),
                        2
                    )


    # ========================================================
    # TRACKING COUNTER
    # ========================================================

    cv2.putText(
        frame,
        f"TRACKED: {len(current_ids)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (195, 230, 0),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    video_display.image(
        rgb,
        channels="RGB",
        use_container_width=True
    )


    # ========================================================
    # METRICS
    # ========================================================

    with tracked_box:

        metric_card(
            "Currently Tracked",
            len(current_ids),
            "◉",
            True
        )


    with cars_box:

        metric_card(
            "Cars",
            cars,
            "🚗"
        )


    with bikes_box:

        metric_card(
            "Motorcycles",
            motorcycles,
            "🏍"
        )


    with buses_box:

        metric_card(
            "Buses",
            buses,
            "🚌"
        )


    with trucks_box:

        metric_card(
            "Trucks",
            trucks,
            "🚚"
        )


    # ========================================================
    # PLATE TABLE
    # ========================================================

    if plate_data:

        plate_df = create_plate_dataframe(
            plate_data
        )

        plate_display.dataframe(
            plate_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Vehicle ID":
                    st.column_config.TextColumn(
                        "Vehicle ID"
                    ),

                "License Plate":
                    st.column_config.TextColumn(
                        "License Plate"
                    ),

                "Confidence":
                    st.column_config.NumberColumn(
                        "Confidence",
                        format="%.2f"
                    )
            }
        )

    else:

        plate_display.info(
            "No license plates recognized yet."
        )


    # ========================================================
    # PROGRESS
    # ========================================================

    if total_frames > 0:

        progress.progress(
            min(
                frame_number /
                total_frames,
                1.0
            )
        )


    # ========================================================
    # STATUS
    # ========================================================

    status_display.markdown(
        f"""
        <div class="status-bar">

            <span class="status-active">
                ● PROCESSING
            </span>

            <span>
                FRAME {frame_number}/{total_frames}
            </span>

            <span>·</span>

            <span>
                TRACK IDS: {len(all_ids)}
            </span>

            <span>·</span>

            <span>
                PLATES: {len(plate_data)}
            </span>

            <span>·</span>

            <span>
                FPS: {fps:.0f}
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RELEASE
# ============================================================

cap.release()

progress.progress(1.0)


# ============================================================
# COMPLETED
# ============================================================

st.success(
    "🎉 AI Traffic Analysis Completed!"
)


# ============================================================
# FINAL RESULTS
# ============================================================

section_header(
    "Final Results",
    "ANALYSIS SUMMARY"
)

a, b, c = st.columns(3)


with a:

    metric_card(
        "Track IDs Observed",
        len(all_ids),
        "◉",
        True
    )


with b:

    metric_card(
        "License Plates Recognized",
        len(plate_data),
        "▣"
    )


with c:

    metric_card(
        "Frames Processed",
        processed_frames,
        "▤"
    )


# ============================================================
# FINAL PLATES
# ============================================================

section_header(
    "Final License Plate Results",
    "OCR SUMMARY"
)


if plate_data:

    final_df = create_plate_dataframe(
        plate_data
    )

    st.dataframe(
        final_df,
        use_container_width=True,
        hide_index=True,
        column_config={

            "Vehicle ID":
                st.column_config.TextColumn(
                    "Vehicle ID"
                ),

            "License Plate":
                st.column_config.TextColumn(
                    "License Plate"
                ),

            "Confidence":
                st.column_config.NumberColumn(
                    "Confidence",
                    format="%.2f"
                )
        }
    )

else:

    st.info(
        "No license plates were recognized."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">

        AI TRAFFIC INTELLIGENCE
        &nbsp;·&nbsp;
        COMPUTER VISION ANALYTICS
        &nbsp;·&nbsp;
        YOLO11 + BYTETRACK + OCR

    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import cv2
import re
import pandas as pd
from ultralytics import YOLO
import easyocr


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# THEME
# =========================================================

def inject_theme():

    st.markdown(
        """
        <style>

        @import url(
            'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap'
        );

        :root {
            --bg: #070B0F;
            --panel: #0D1319;
            --panel-2: #111920;
            --border: #202B35;
            --border-light: #2B3945;
            --text: #E8EEF3;
            --muted: #7F8D99;
            --cyan: #00E6C3;
            --blue: #4DA3FF;
            --orange: #FFB454;
        }

        html,
        body,
        [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 5% 0%,
                    rgba(0,230,195,0.07),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 95% 10%,
                    rgba(77,163,255,0.045),
                    transparent 25%
                ),
                #070B0F;
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        footer {
            visibility: hidden;
        }

        #MainMenu {
            visibility: hidden;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }


        /* =====================================================
           HERO HEADER
           ===================================================== */

        .hero {
            position: relative;
            padding: 25px 28px;
            margin-bottom: 22px;

            background:
                linear-gradient(
                    135deg,
                    rgba(17,25,32,0.98),
                    rgba(10,15,20,0.98)
                );

            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;

            box-shadow:
                0 20px 60px rgba(0,0,0,0.25);
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

            opacity: 0.75;
        }

        .hero-grid {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 20px;
        }

        .hero-title {
            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 26px;
            font-weight: 700;
            letter-spacing: 0.5px;

            color: var(--text);
        }

        .hero-subtitle {
            margin-top: 8px;

            font-size: 13px;

            color: var(--muted);

            letter-spacing: 0.3px;
        }

        .hero-tech {
            margin-top: 12px;

            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .tech {
            padding: 4px 9px;

            border: 1px solid var(--border-light);
            border-radius: 5px;

            background:
                rgba(255,255,255,0.02);

            color: #9EABB6;

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 9px;
            letter-spacing: 0.8px;
        }

        .system-status {
            display: inline-flex;

            align-items: center;
            gap: 9px;

            padding: 8px 13px;

            border-radius: 20px;

            border: 1px solid
                rgba(0,230,195,0.28);

            background:
                rgba(0,230,195,0.06);

            color: var(--cyan);

            font-family:
                'JetBrains Mono',
                monospace;

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
                0 0 12px rgba(0,230,195,0.8);
        }


        /* =====================================================
           SECTION HEADER
           ===================================================== */

        .section-header {
            display: flex;

            align-items: center;
            gap: 10px;

            margin-top: 28px;
            margin-bottom: 12px;
        }

        .section-line {
            width: 3px;
            height: 18px;

            border-radius: 2px;

            background: var(--cyan);

            box-shadow:
                0 0 10px rgba(0,230,195,0.35);
        }

        .section-title {
            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 11px;
            font-weight: 600;

            letter-spacing: 1.7px;

            text-transform: uppercase;

            color: #8D9AA6;
        }

        .section-meta {
            margin-left: auto;

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 9px;

            color: #53616D;

            letter-spacing: 0.8px;
        }


        /* =====================================================
           METRIC CARDS
           ===================================================== */

        .metric-card {
            min-height: 102px;

            padding: 17px 18px;

            background:
                linear-gradient(
                    145deg,
                    rgba(17,25,32,0.96),
                    rgba(11,16,21,0.96)
                );

            border: 1px solid var(--border);

            border-radius: 9px;

            position: relative;

            overflow: hidden;

            transition:
                border-color .2s ease,
                transform .2s ease;
        }

        .metric-card:hover {
            border-color:
                rgba(0,230,195,0.35);

            transform: translateY(-1px);
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
            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 9px;
            font-weight: 500;

            letter-spacing: 1.2px;

            text-transform: uppercase;

            color: #687782;
        }

        .metric-value {
            margin-top: 10px;

            font-family:
                'JetBrains Mono',
                monospace;

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
            float: right;

            font-size: 16px;

            opacity: 0.55;
        }


        /* =====================================================
           UPLOAD PANEL
           ===================================================== */

        .upload-panel {
            padding: 18px;

            background:
                rgba(13,19,25,0.85);

            border: 1px solid var(--border);

            border-radius: 9px;
        }

        .upload-title {
            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 11px;
            font-weight: 600;

            letter-spacing: 1px;

            color: var(--text);
        }

        .upload-description {
            margin-top: 5px;

            font-size: 11px;

            color: var(--muted);
        }

        [data-testid="stFileUploader"] {
            margin-top: 12px;

            background:
                rgba(255,255,255,0.015);

            border: 1px dashed #33414D;

            border-radius: 8px;

            padding: 14px;
        }

        [data-testid="stFileUploader"]:hover {
            border-color:
                rgba(0,230,195,0.45);
        }

        [data-testid="stFileUploaderDropzone"] {
            background: transparent;
        }


        /* =====================================================
           BUTTON
           ===================================================== */

        .stButton > button {
            width: 100%;

            min-height: 44px;

            margin-top: 10px;

            border:
                1px solid rgba(0,230,195,0.55);

            border-radius: 7px;

            background:
                linear-gradient(
                    135deg,
                    rgba(0,230,195,0.13),
                    rgba(0,230,195,0.045)
                );

            color: var(--cyan);

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 11px;
            font-weight: 600;

            letter-spacing: 0.7px;

            transition: all .2s ease;
        }

        .stButton > button:hover {
            border-color: var(--cyan);

            background:
                rgba(0,230,195,0.14);

            box-shadow:
                0 0 20px rgba(0,230,195,0.08);

            color: var(--cyan);
        }


        /* =====================================================
           LIVE FEED
           ===================================================== */

        .feed-container {
            position: relative;

            padding: 7px;

            background: #090E13;

            border: 1px solid var(--border);

            border-radius: 10px;

            box-shadow:
                inset 0 0 30px rgba(0,0,0,0.35);
        }

        .live-badge {
            display: inline-flex;

            align-items: center;
            gap: 7px;

            padding: 5px 9px;

            margin-bottom: 8px;

            border-radius: 4px;

            background:
                rgba(7,11,15,0.85);

            border: 1px solid
                rgba(255,70,70,0.3);

            color: #FF7777;

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 9px;
            font-weight: 600;

            letter-spacing: 1px;
        }

        .live-dot {
            width: 6px;
            height: 6px;

            border-radius: 50%;

            background: #FF5555;

            box-shadow:
                0 0 9px rgba(255,70,70,0.7);
        }

        [data-testid="stImage"] {
            width: 100%;

            padding: 0;

            background: #080C10;

            border: 0;

            border-radius: 6px;

            overflow: hidden;
        }

        [data-testid="stImage"] img {
            border-radius: 5px;
        }


        /* =====================================================
           PLATE TABLE
           ===================================================== */

        .plate-panel {
            background:
                rgba(13,19,25,0.9);

            border: 1px solid var(--border);

            border-radius: 9px;

            padding: 5px;
        }

        [data-testid="stDataFrame"] {
            border:
                1px solid var(--border);

            border-radius: 7px;

            overflow: hidden;
        }


        /* =====================================================
           STATUS BAR
           ===================================================== */

        .status-bar {
            display: flex;

            align-items: center;
            gap: 8px;

            margin-top: 9px;

            padding: 9px 12px;

            background:
                rgba(13,19,25,0.75);

            border: 1px solid var(--border);

            border-radius: 6px;

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 9px;

            color: #697883;

            letter-spacing: 0.6px;
        }

        .status-active {
            color: var(--cyan);
        }


        /* =====================================================
           EMPTY STATE
           ===================================================== */

        .empty-state {
            padding: 38px 20px;

            text-align: center;

            background:
                rgba(13,19,25,0.55);

            border:
                1px dashed #293640;

            border-radius: 9px;

            color: #667581;

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 10px;

            letter-spacing: 1px;
        }

        .empty-icon {
            font-size: 27px;

            margin-bottom: 10px;

            opacity: 0.55;
        }


        /* =====================================================
           ALERTS
           ===================================================== */

        .stAlert {
            border-radius: 7px !important;
            border: 1px solid var(--border) !important;
        }


        /* =====================================================
           PROGRESS
           ===================================================== */

        [data-testid="stProgress"] {
            margin-top: 10px;
        }

        [data-testid="stProgressBar"] {
            background: #151D24;
        }


        /* =====================================================
           MOBILE
           ===================================================== */

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

        }

        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HEADER
# =========================================================

def render_header():

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

                        <span class="tech">
                            YOLO11
                        </span>

                        <span class="tech">
                            BYTE TRACK
                        </span>

                        <span class="tech">
                            PLATE DETECTION
                        </span>

                        <span class="tech">
                            EASYOCR
                        </span>

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


# =========================================================
# SECTION HEADER
# =========================================================

def section_header(title, meta=""):

    meta_html = ""

    if meta:

        meta_html = f"""
            <div class="section-meta">
                {meta}
            </div>
        """

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


# =========================================================
# KPI
# =========================================================

def kpi_html(
    label,
    value,
    accent=False,
    icon=""
):

    value_class = (
        "metric-value accent"
        if accent
        else
        "metric-value"
    )

    return f"""
    <div class="metric-card">

        <span class="metric-icon">
            {icon}
        </span>

        <div class="metric-label">
            {label}
        </div>

        <div class="{value_class}">
            {value}
        </div>

    </div>
    """


# =========================================================
# INITIAL UI
# =========================================================

inject_theme()

render_header()


# =========================================================
# LOAD MODELS
# =========================================================

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


# =========================================================
# OCR CLEANING
# =========================================================

def clean_text(text):

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# =========================================================
# INPUT FEED
# =========================================================

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

    </div>
    """,
    unsafe_allow_html=True
)


uploaded = st.file_uploader(
    "Upload Traffic Video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ],
    label_visibility="collapsed"
)


# =========================================================
# NO VIDEO
# =========================================================

if uploaded is None:

    st.markdown(
        """
        <div class="empty-state">

            <div class="empty-icon">
                🎥
            </div>

            AWAITING VIDEO INPUT

            <div style="
                margin-top:8px;
                font-family:IBM Plex Sans,sans-serif;
                font-size:11px;
                letter-spacing:0;
                color:#53616D;
            ">
                Upload traffic footage to initialize
                the AI detection pipeline
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# =========================================================
# FILE SIZE
# =========================================================

file_size_mb = uploaded.size / (
    1024 * 1024
)


if file_size_mb > 200:

    st.error(
        "Video exceeds the 200 MB limit."
    )

    st.stop()


st.markdown(
    f"""
    <div class="status-bar">

        <span class="status-active">
            ● FILE READY
        </span>

        <span>
            {uploaded.name}
        </span>

        <span>
            ·
        </span>

        <span>
            {file_size_mb:.1f} MB
        </span>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# START BUTTON
# =========================================================

start = st.button(
    "🚀  START AI TRAFFIC ANALYSIS",
    type="primary"
)


if not start:

    st.stop()


# =========================================================
# SAVE VIDEO
# =========================================================

video_path = "uploaded_traffic.mp4"


with open(
    video_path,
    "wb"
) as f:

    f.write(
        uploaded.getbuffer()
    )


# =========================================================
# MODEL LOADING
# =========================================================

section_header(
    "AI Engine",
    "INITIALIZATION"
)

with st.spinner(
    "Loading YOLO11, plate detector and OCR engine..."
):

    (
        vehicle_model,
        plate_model,
        reader
    ) = load_models()


st.success(
    "AI inference engine initialized successfully."
)


# =========================================================
# OPEN VIDEO
# =========================================================

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


duration = (
    total_frames / fps
    if total_frames > 0
    else 0
)


# =========================================================
# PROCESSING VARIABLES
# =========================================================

frame_number = 0

processed_frames = 0

all_ids = set()

plate_data = {}

last_ocr = {}


FRAME_SKIP = 3

OCR_INTERVAL = 45


# =========================================================
# LIVE FEED
# =========================================================

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

status = st.empty()


# =========================================================
# LIVE STATISTICS
# =========================================================

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


# =========================================================
# LICENSE PLATES
# =========================================================

section_header(
    "License Plates",
    "OCR DETECTIONS"
)


plate_display = st.empty()


# =========================================================
# VIDEO PROCESSING
# =========================================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame_number += 1


    if frame_number % FRAME_SKIP != 0:

        continue


    processed_frames += 1


    # =====================================================
    # VEHICLE TRACKING
    # =====================================================

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


    # =====================================================
    # VEHICLE DETECTIONS
    # =====================================================

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


            for (
                vehicle_id,
                cls,
                box
            ) in zip(
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


                # =============================================
                # VEHICLE TYPE
                # =============================================

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


                # =============================================
                # BOX
                # =============================================

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


                # =============================================
                # OCR
                # =============================================

                should_ocr = (

                    vehicle_id
                    not in last_ocr

                    or

                    frame_number
                    -
                    last_ocr[
                        vehicle_id
                    ]
                    >= OCR_INTERVAL

                )


                if (
                    should_ocr
                    and
                    crop.size > 0
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
                                        best_conf
                                        >
                                        previous[
                                            "confidence"
                                        ]
                                    ):

                                        plate_data[
                                            vehicle_id
                                        ] = {

                                            "text":
                                                best_text,

                                            "confidence":
                                                best_conf

                                        }


                    except Exception:

                        pass


                # =============================================
                # VEHICLE BOX
                # =============================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (195, 230, 0),
                    2
                )


                # =============================================
                # VEHICLE LABEL
                # =============================================

                label = (
                    f"{vehicle_type} "
                    f"ID:{vehicle_id}"
                )


                cv2.putText(
                    frame,
                    label,
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


                # =============================================
                # PLATE LABEL
                # =============================================

                if vehicle_id in plate_data:

                    plate_text = (
                        plate_data[
                            vehicle_id
                        ][
                            "text"
                        ]
                    )


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


    # =====================================================
    # TRACKING TEXT
    # =====================================================

    cv2.putText(
        frame,
        f"TRACKED: {len(current_ids)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (195, 230, 0),
        2
    )


    # =====================================================
    # DISPLAY FRAME
    # =====================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    video_display.image(
        rgb,
        channels="RGB",
        use_container_width=True
    )


    # =====================================================
    # LIVE KPI CARDS
    # =====================================================

    tracked_box.markdown(
        kpi_html(
            "Currently Tracked",
            len(current_ids),
            True,
            "◉"
        ),
        unsafe_allow_html=True
    )


    cars_box.markdown(
        kpi_html(
            "Cars",
            cars,
            False,
            "🚗"
        ),
        unsafe_allow_html=True
    )


    bikes_box.markdown(
        kpi_html(
            "Motorcycles",
            motorcycles,
            False,
            "🏍"
        ),
        unsafe_allow_html=True
    )


    buses_box.markdown(
        kpi_html(
            "Buses",
            buses,
            False,
            "🚌"
        ),
        unsafe_allow_html=True
    )


    trucks_box.markdown(
        kpi_html(
            "Trucks",
            trucks,
            False,
            "🚚"
        ),
        unsafe_allow_html=True
    )


    # =====================================================
    # PLATE TABLE
    # =====================================================

    if plate_data:

        rows = []


        for (
            vehicle_id,
            data
        ) in plate_data.items():

            rows.append({

                "Vehicle ID":
                    f"#{vehicle_id}",

                "License Plate":
                    data["text"],

                "Confidence":
                    round(
                        data["confidence"],
                        2
                    )

            })


        plate_df = pd.DataFrame(
            rows
        )


        plate_df["_sort"] = (
            plate_df[
                "Vehicle ID"
            ]
            .str.replace(
                "#",
                "",
                regex=False
            )
            .astype(int)
        )


        plate_df = (
            plate_df
            .sort_values("_sort")
            .drop(columns="_sort")
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


    # =====================================================
    # PROGRESS
    # =====================================================

    if total_frames > 0:

        progress.progress(
            min(
                frame_number /
                total_frames,
                1.0
            )
        )


    # =====================================================
    # STATUS
    # =====================================================

    elapsed_frames = processed_frames

    status.markdown(
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


# =========================================================
# RELEASE VIDEO
# =========================================================

cap.release()

progress.progress(1.0)


# =========================================================
# COMPLETED
# =========================================================

st.success(
    "🎉 AI Traffic Analysis Completed!"
)


# =========================================================
# FINAL RESULTS
# =========================================================

section_header(
    "Final Results",
    "ANALYSIS SUMMARY"
)


a, b, c = st.columns(3)


a.markdown(
    kpi_html(
        "Track IDs Observed",
        len(all_ids),
        True,
        "◉"
    ),
    unsafe_allow_html=True
)


b.markdown(
    kpi_html(
        "License Plates Recognized",
        len(plate_data),
        False,
        "▣"
    ),
    unsafe_allow_html=True
)


c.markdown(
    kpi_html(
        "Frames Processed",
        processed_frames,
        False,
        "▤"
    ),
    unsafe_allow_html=True
)


# =========================================================
# FINAL LICENSE PLATE RESULTS
# =========================================================

section_header(
    "Final License Plate Results",
    "OCR SUMMARY"
)


if plate_data:

    final_rows = []


    for (
        vehicle_id,
        data
    ) in plate_data.items():

        final_rows.append({

            "Vehicle ID":
                f"#{vehicle_id}",

            "License Plate":
                data["text"],

            "Confidence":
                round(
                    data["confidence"],
                    2
                )

        })


    final_df = pd.DataFrame(
        final_rows
    )


    final_df["_sort"] = (
        final_df[
            "Vehicle ID"
        ]
        .str.replace(
            "#",
            "",
            regex=False
        )
        .astype(int)
    )


    final_df = (
        final_df
        .sort_values("_sort")
        .drop(columns="_sort")
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


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        margin-top:45px;
        padding-top:18px;
        border-top:1px solid #202B35;
        text-align:center;
        font-family:JetBrains Mono,monospace;
        font-size:9px;
        letter-spacing:1px;
        color:#46535E;
    ">

        AI TRAFFIC INTELLIGENCE
        &nbsp;·&nbsp;
        COMPUTER VISION ANALYTICS
        &nbsp;·&nbsp;
        YOLO11 + BYTETRACK + OCR

    </div>
    """,
    unsafe_allow_html=True
)

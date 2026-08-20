import streamlit as st
import cv2
import re
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
# GLOBAL CSS
# ============================================================

def inject_css():

    st.markdown(
        """
        <style>

        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');


        /* =====================================================
           GLOBAL
           ===================================================== */

        html,
        body,
        [class*="css"] {
            font-family: "IBM Plex Sans", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 0% 0%,
                    rgba(0, 230, 195, 0.07),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 100% 0%,
                    rgba(58, 134, 255, 0.045),
                    transparent 25%
                ),
                #070B0F;
            color: #E8EEF3;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        footer {
            display: none !important;
        }

        #MainMenu {
            visibility: hidden !important;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }


        /* =====================================================
           HERO
           ===================================================== */

        .hero-card {
            position: relative;

            padding: 27px 30px;

            margin-bottom: 30px;

            border: 1px solid #202B35;

            border-radius: 12px;

            background:
                linear-gradient(
                    135deg,
                    #101820,
                    #0B1117
                );

            box-shadow:
                0 18px 55px rgba(0,0,0,0.25);

            overflow: hidden;
        }

        .hero-card:before {
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
                    #00E6C3,
                    transparent
                );
        }

        .hero-grid {
            display: flex;

            justify-content: space-between;

            align-items: flex-start;

            gap: 25px;
        }

        .hero-title {
            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 26px;

            font-weight: 700;

            letter-spacing: 0.5px;

            color: #E8EEF3;
        }

        .hero-subtitle {
            margin-top: 9px;

            font-size: 13px;

            color: #7F8D99;

            letter-spacing: 0.3px;
        }

        .tech-row {
            display: flex;

            gap: 7px;

            flex-wrap: wrap;

            margin-top: 15px;
        }

        .tech-badge {
            display: inline-block;

            padding: 5px 9px;

            border: 1px solid #2A3742;

            border-radius: 5px;

            background: rgba(255,255,255,0.02);

            color: #8E9CA7;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 9px;

            font-weight: 500;

            letter-spacing: 0.8px;
        }

        .online-badge {
            display: inline-flex;

            align-items: center;

            padding: 8px 13px;

            border: 1px solid
                rgba(0,230,195,0.30);

            border-radius: 20px;

            background:
                rgba(0,230,195,0.06);

            color: #00E6C3;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 10px;

            font-weight: 600;

            letter-spacing: 1px;

            white-space: nowrap;
        }


        /* =====================================================
           SECTION HEADER
           ===================================================== */

        .section-header {
            display: flex;

            align-items: center;

            gap: 10px;

            margin-top: 30px;

            margin-bottom: 14px;
        }

        .section-marker {
            width: 3px;

            height: 18px;

            border-radius: 3px;

            background: #00E6C3;

            box-shadow:
                0 0 12px
                rgba(0,230,195,0.45);
        }

        .section-title {
            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 11px;

            font-weight: 600;

            letter-spacing: 1.7px;

            text-transform: uppercase;

            color: #8997A3;
        }

        .section-meta {
            margin-left: auto;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 9px;

            letter-spacing: 0.8px;

            color: #4E5D68;
        }


        /* =====================================================
           UPLOAD PANEL
           ===================================================== */

        .upload-panel {
            padding: 20px;

            border: 1px solid #202B35;

            border-radius: 10px;

            background:
                linear-gradient(
                    145deg,
                    rgba(15,22,29,0.96),
                    rgba(10,15,20,0.96)
                );
        }

        .upload-heading {
            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 11px;

            font-weight: 600;

            letter-spacing: 1px;

            color: #E8EEF3;
        }

        .upload-text {
            margin-top: 7px;

            font-size: 11px;

            line-height: 1.6;

            color: #788792;
        }

        .upload-limit {
            margin-top: 13px;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 9px;

            letter-spacing: 0.8px;

            color: #53616D;
        }


        /* =====================================================
           STREAMLIT UPLOADER
           ===================================================== */

        [data-testid="stFileUploader"] {
            margin-top: 12px;
        }

        [data-testid="stFileUploaderDropzone"] {
            min-height: 125px;

            background:
                rgba(255,255,255,0.012) !important;

            border: 1px dashed #34434F !important;

            border-radius: 9px !important;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color:
                #00E6C3 !important;

            background:
                rgba(0,230,195,0.025) !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] {
            color: #74838E !important;
        }


        /* =====================================================
           BUTTON
           ===================================================== */

        .stButton {
            margin-top: 12px;
        }

        .stButton > button {
            width: 100%;

            min-height: 46px;

            border-radius: 7px;

            border: 1px solid
                rgba(0,230,195,0.50);

            background:
                linear-gradient(
                    135deg,
                    rgba(0,230,195,0.13),
                    rgba(0,230,195,0.035)
                );

            color: #00E6C3;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 11px;

            font-weight: 600;

            letter-spacing: 0.7px;
        }

        .stButton > button:hover {
            border-color: #00E6C3;

            background:
                rgba(0,230,195,0.13);

            color: #00E6C3;

            box-shadow:
                0 0 22px
                rgba(0,230,195,0.08);
        }


        /* =====================================================
           FILE READY
           ===================================================== */

        .file-ready {
            display: flex;

            align-items: center;

            gap: 10px;

            margin-top: 12px;

            padding: 10px 13px;

            border: 1px solid #202B35;

            border-radius: 7px;

            background: #0C1319;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 9px;

            color: #64737F;
        }

        .file-ready-dot {
            width: 7px;

            height: 7px;

            border-radius: 50%;

            background: #00E6C3;

            box-shadow:
                0 0 8px
                rgba(0,230,195,0.7);
        }

        .file-name {
            color: #A8B4BD;
        }


        /* =====================================================
           EMPTY STATE
           ===================================================== */

        .empty-state {
            margin-top: 12px;

            padding: 40px 20px;

            text-align: center;

            border: 1px dashed #293741;

            border-radius: 9px;

            background:
                rgba(12,18,24,0.55);

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 10px;

            font-weight: 600;

            letter-spacing: 1px;

            color: #667681;
        }

        .empty-icon {
            margin-bottom: 10px;

            font-size: 29px;

            opacity: 0.55;
        }

        .empty-description {
            margin-top: 9px;

            font-family:
                "IBM Plex Sans",
                sans-serif;

            font-size: 11px;

            font-weight: 400;

            letter-spacing: 0;

            color: #53616D;
        }


        /* =====================================================
           LIVE FEED
           ===================================================== */

        .live-container {
            padding: 7px;

            border: 1px solid #202B35;

            border-radius: 10px;

            background: #080D12;

            box-shadow:
                inset 0 0 35px
                rgba(0,0,0,0.35);
        }

        .live-label {
            display: inline-flex;

            align-items: center;

            gap: 7px;

            margin: 2px 0 8px 2px;

            padding: 5px 9px;

            border: 1px solid
                rgba(255,80,80,0.25);

            border-radius: 4px;

            background:
                rgba(255,60,60,0.035);

            color: #FF7777;

            font-family:
                "JetBrains Mono",
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
                0 0 9px
                rgba(255,70,70,0.8);
        }

        [data-testid="stImage"] {
            padding: 0 !important;

            border: 0 !important;

            background: transparent !important;
        }

        [data-testid="stImage"] img {
            border-radius: 6px;
        }


        /* =====================================================
           METRIC CARDS
           ===================================================== */

        .metric-card {
            position: relative;

            min-height: 100px;

            padding: 17px;

            border: 1px solid #202B35;

            border-radius: 9px;

            background:
                linear-gradient(
                    145deg,
                    #101820,
                    #0C1218
                );

            overflow: hidden;
        }

        .metric-card:after {
            content: "";

            position: absolute;

            right: -25px;

            bottom: -35px;

            width: 90px;

            height: 90px;

            border-radius: 50%;

            background:
                rgba(0,230,195,0.035);
        }

        .metric-icon {
            float: right;

            font-size: 16px;

            opacity: 0.45;
        }

        .metric-label {
            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 9px;

            letter-spacing: 1.1px;

            text-transform: uppercase;

            color: #687883;
        }

        .metric-value {
            margin-top: 9px;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 27px;

            font-weight: 600;

            color: #E8EEF3;
        }

        .metric-value.accent {
            color: #00E6C3;

            text-shadow:
                0 0 18px
                rgba(0,230,195,0.16);
        }


        /* =====================================================
           TABLE
           ===================================================== */

        [data-testid="stDataFrame"] {
            border: 1px solid #202B35;

            border-radius: 8px;

            overflow: hidden;
        }


        /* =====================================================
           STATUS BAR
           ===================================================== */

        .status-bar {
            display: flex;

            align-items: center;

            flex-wrap: wrap;

            gap: 9px;

            margin-top: 9px;

            padding: 9px 12px;

            border: 1px solid #202B35;

            border-radius: 6px;

            background: #0B1117;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 9px;

            letter-spacing: 0.5px;

            color: #62717D;
        }

        .status-green {
            color: #00E6C3;
        }


        /* =====================================================
           MOBILE
           ===================================================== */

        @media(max-width: 800px) {

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero-grid {
                flex-direction: column;
            }

            .online-badge {
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
# HERO
# ============================================================

def render_hero():

    st.markdown(
        """
        <div class="hero-card">

            <div class="hero-grid">

                <div>

                    <div class="hero-title">
                        🚦 AI TRAFFIC INTELLIGENCE
                    </div>

                    <div class="hero-subtitle">
                        Real-time vehicle detection, tracking
                        and license plate recognition
                    </div>

                    <div class="tech-row">

                        <span class="tech-badge">
                            YOLO11
                        </span>

                        <span class="tech-badge">
                            BYTE TRACK
                        </span>

                        <span class="tech-badge">
                            PLATE DETECTION
                        </span>

                        <span class="tech-badge">
                            EASYOCR
                        </span>

                    </div>

                </div>

                <div class="online-badge">
                    ●&nbsp; SYSTEM ONLINE
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SECTION
# ============================================================

def section(title, meta=""):

    st.markdown(
        f"""
        <div class="section-header">

            <div class="section-marker"></div>

            <div class="section-title">
                {title}
            </div>

            <div class="section-meta">
                {meta}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# METRIC
# ============================================================

def metric(label, value, icon="", accent=False):

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


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():

    vehicle_model = YOLO("yolo11n.pt")

    plate_model = YOLO("models/best.pt")

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
# OCR CLEANING
# ============================================================

def clean_text(text):

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# START UI
# ============================================================

inject_css()

render_hero()


# ============================================================
# INPUT
# ============================================================

section(
    "Input Feed",
    "VIDEO ANALYSIS PIPELINE"
)

st.markdown(
    """
    <div class="upload-panel">

        <div class="upload-heading">
            TRAFFIC VIDEO SOURCE
        </div>

        <div class="upload-text">
            Upload MP4, AVI, MOV or MKV footage for
            AI-powered traffic analysis.
            Maximum file size: 200 MB.
        </div>

        <div class="upload-limit">
            200 MB PER FILE &nbsp;·&nbsp;
            MP4 &nbsp;·&nbsp;
            AVI &nbsp;·&nbsp;
            MOV &nbsp;·&nbsp;
            MKV
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded = st.file_uploader(
    "Traffic video",
    type=["mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed"
)


# ============================================================
# EMPTY
# ============================================================

if uploaded is None:

    st.markdown(
        """
        <div class="empty-state">

            <div class="empty-icon">
                🎥
            </div>

            AWAITING VIDEO INPUT

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

file_size_mb = uploaded.size / (1024 * 1024)

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
    <div class="file-ready">

        <span class="file-ready-dot"></span>

        <span class="file-name">
            {uploaded.name}
        </span>

        <span>
            ·
        </span>

        <span>
            {file_size_mb:.1f} MB
        </span>

        <span>
            ·
        </span>

        <span style="color:#00E6C3;">
            READY
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
    type="primary"
)

if not start:

    st.stop()


# ============================================================
# SAVE VIDEO
# ============================================================

video_path = "uploaded_traffic.mp4"

with open(video_path, "wb") as f:

    f.write(
        uploaded.getbuffer()
    )


# ============================================================
# AI ENGINE
# ============================================================

section(
    "AI Engine",
    "MODEL INITIALIZATION"
)

with st.spinner(
    "Initializing YOLO11, ByteTrack and EasyOCR..."
):

    (
        vehicle_model,
        plate_model,
        reader
    ) = load_models()


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    st.error(
        "Could not open the uploaded video."
    )

    st.stop()


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30


# ============================================================
# VARIABLES
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

section(
    "Live Feed",
    "REAL-TIME INFERENCE"
)

st.markdown(
    """
    <div class="live-label">
        <span class="live-dot"></span>
        LIVE INFERENCE
    </div>
    """,
    unsafe_allow_html=True
)

video_display = st.empty()

progress = st.progress(0)

status = st.empty()


# ============================================================
# STATISTICS
# ============================================================

section(
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

section(
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


    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    results = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.40,
        iou=0.50,
        verbose=False
    )

    result = results[0]

    current_ids = set()

    cars = 0
    motorcycles = 0
    buses = 0
    trucks = 0


    # --------------------------------------------------------
    # DETECTIONS
    # --------------------------------------------------------

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

                current_ids.add(vehicle_id)

                all_ids.add(vehicle_id)


                # ------------------------------------------------
                # TYPE
                # ------------------------------------------------

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


                # ------------------------------------------------
                # COORDINATES
                # ------------------------------------------------

                x1, y1, x2, y2 = box

                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(
                    frame.shape[1],
                    x2
                )

                y2 = min(
                    frame.shape[0],
                    y2
                )

                if x2 <= x1 or y2 <= y1:
                    continue


                crop = frame[
                    y1:y2,
                    x1:x2
                ]


                # ------------------------------------------------
                # OCR
                # ------------------------------------------------

                should_ocr = (
                    vehicle_id not in last_ocr
                    or
                    frame_number -
                    last_ocr[vehicle_id]
                    >= OCR_INTERVAL
                )


                if should_ocr and crop.size > 0:

                    last_ocr[vehicle_id] = frame_number

                    try:

                        plate_results = plate_model(
                            crop,
                            conf=0.50,
                            verbose=False
                        )

                        for plate_result in plate_results:

                            if plate_result.boxes is None:
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

                                px1 = max(0, px1)
                                py1 = max(0, py1)

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


                # ------------------------------------------------
                # DRAW VEHICLE BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (195, 230, 0),
                    2
                )


                # ------------------------------------------------
                # VEHICLE LABEL
                # ------------------------------------------------

                cv2.putText(
                    frame,
                    f"{vehicle_type} ID:{vehicle_id}",
                    (
                        x1,
                        max(25, y1 - 8)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (195, 230, 0),
                    2
                )


                # ------------------------------------------------
                # PLATE LABEL
                # ------------------------------------------------

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


    # --------------------------------------------------------
    # TRACK COUNT
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"TRACKED: {len(current_ids)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (195, 230, 0),
        2
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    video_display.image(
        rgb,
        channels="RGB",
        use_container_width=True
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    tracked_box.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">◉</div>
            <div class="metric-label">
                Currently Tracked
            </div>
            <div class="metric-value accent">
                {len(current_ids)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    cars_box.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🚗</div>
            <div class="metric-label">
                Cars
            </div>
            <div class="metric-value">
                {cars}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    bikes_box.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🏍</div>
            <div class="metric-label">
                Motorcycles
            </div>
            <div class="metric-value">
                {motorcycles}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    buses_box.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🚌</div>
            <div class="metric-label">
                Buses
            </div>
            <div class="metric-value">
                {buses}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    trucks_box.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🚚</div>
            <div class="metric-label">
                Trucks
            </div>
            <div class="metric-value">
                {trucks}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PLATE TABLE
    # --------------------------------------------------------

    if plate_data:

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

        plate_df = pd.DataFrame(rows)

        plate_df["_sort"] = (
            plate_df["Vehicle ID"]
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


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if total_frames > 0:

        progress.progress(
            min(
                frame_number /
                total_frames,
                1.0
            )
        )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    status.markdown(
        f"""
        <div class="status-bar">

            <span class="status-green">
                ● PROCESSING
            </span>

            <span>
                FRAME {frame_number}/{total_frames}
            </span>

            <span>·</span>

            <span>
                TRACK IDS {len(all_ids)}
            </span>

            <span>·</span>

            <span>
                PLATES {len(plate_data)}
            </span>

            <span>·</span>

            <span>
                FPS {fps:.0f}
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
# COMPLETE
# ============================================================

st.success(
    "AI Traffic Analysis Completed Successfully."
)


# ============================================================
# FINAL RESULTS
# ============================================================

section(
    "Final Results",
    "ANALYSIS SUMMARY"
)

a, b, c = st.columns(3)


with a:

    metric(
        "Track IDs Observed",
        len(all_ids),
        "◉",
        True
    )


with b:

    metric(
        "License Plates Recognized",
        len(plate_data),
        "▣"
    )


with c:

    metric(
        "Frames Processed",
        processed_frames,
        "▤"
    )


# ============================================================
# FINAL PLATES
# ============================================================

section(
    "Final License Plate Results",
    "OCR SUMMARY"
)


if plate_data:

    final_rows = []

    for vehicle_id, data in plate_data.items():

        final_rows.append(
            {
                "Vehicle ID":
                    f"#{vehicle_id}",

                "License Plate":
                    data["text"],

                "Confidence":
                    round(
                        data["confidence"],
                        2
                    )
            }
        )

    final_df = pd.DataFrame(
        final_rows
    )

    final_df["_sort"] = (
        final_df["Vehicle ID"]
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


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:50px;
        padding-top:18px;
        border-top:1px solid #202B35;
        text-align:center;
        font-family:'JetBrains Mono',monospace;
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

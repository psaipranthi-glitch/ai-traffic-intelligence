import streamlit as st
import cv2
import re
import pandas as pd
import os
import time
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
# CUSTOM CSS
# ============================================================

def load_css():

    st.markdown(
        """
        <style>

        @import url(
            'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap'
        );

        /* ================================
           GLOBAL
        ================================= */

        .stApp {
            background:
                radial-gradient(
                    circle at 10% 0%,
                    rgba(0,230,195,0.065),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 90% 15%,
                    rgba(50,130,255,0.045),
                    transparent 25%
                ),
                #070B0F;
        }

        html,
        body,
        [class*="css"] {
            font-family: "IBM Plex Sans", sans-serif;
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


        /* ================================
           HEADER
        ================================= */

        .hero-container {
            background:
                linear-gradient(
                    135deg,
                    #111920,
                    #0B1117
                );

            border: 1px solid #24313B;
            border-radius: 12px;

            padding: 26px 30px;

            margin-bottom: 30px;

            box-shadow:
                0 15px 45px rgba(0,0,0,0.25);
        }

        .hero-container::before {
            content: "";

            display: block;

            height: 2px;

            margin: -26px -30px 22px -30px;

            background:
                linear-gradient(
                    90deg,
                    transparent,
                    #00E6C3,
                    transparent
                );
        }

        .hero-title {
            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 27px;

            font-weight: 700;

            letter-spacing: 0.5px;

            color: #E8EEF3;

            margin-bottom: 7px;
        }

        .hero-subtitle {
            color: #82919D;

            font-size: 13px;

            letter-spacing: 0.2px;
        }

        .tech-badge {
            display: inline-block;

            margin-top: 14px;

            margin-right: 6px;

            padding: 5px 9px;

            border: 1px solid #2B3944;

            border-radius: 5px;

            color: #8998A4;

            background: rgba(255,255,255,0.02);

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 9px;

            letter-spacing: 0.8px;
        }

        .online-badge {
            margin-top: 8px;

            padding: 8px 13px;

            border-radius: 20px;

            border: 1px solid rgba(0,230,195,0.3);

            background: rgba(0,230,195,0.06);

            color: #00E6C3;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 10px;

            font-weight: 600;

            letter-spacing: 1px;

            text-align: center;
        }


        /* ================================
           SECTION
        ================================= */

        .section-title {
            margin-top: 30px;

            margin-bottom: 4px;

            padding-left: 10px;

            border-left: 3px solid #00E6C3;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 12px;

            font-weight: 600;

            letter-spacing: 1.5px;

            text-transform: uppercase;

            color: #A0ADB7;
        }

        .section-subtitle {
            margin-left: 13px;

            margin-bottom: 15px;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 9px;

            letter-spacing: 0.8px;

            color: #53616D;
        }


        /* ================================
           UPLOAD CARD
        ================================= */

        .upload-card {
            background: #0D1319;

            border: 1px solid #25323D;

            border-radius: 10px;

            padding: 20px;

            margin-bottom: 10px;
        }

        .upload-heading {
            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 12px;

            font-weight: 600;

            color: #E8EEF3;

            letter-spacing: 1px;
        }

        .upload-text {
            color: #7D8B97;

            font-size: 11px;

            margin-top: 5px;

            line-height: 1.6;
        }


        /* ================================
           FILE UPLOADER
        ================================= */

        [data-testid="stFileUploader"] {
            background: #0D1319;

            border: 1px dashed #33434F;

            border-radius: 9px;

            padding: 12px;

            margin-top: 8px;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #00E6C3;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: transparent;
        }


        /* ================================
           BUTTON
        ================================= */

        .stButton > button {
            width: 100%;

            min-height: 46px;

            margin-top: 10px;

            border-radius: 7px;

            border: 1px solid #00A98F;

            background:
                linear-gradient(
                    135deg,
                    rgba(0,230,195,0.15),
                    rgba(0,230,195,0.05)
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
                rgba(0,230,195,0.14);

            color: #00E6C3;

            box-shadow:
                0 0 22px rgba(0,230,195,0.08);
        }


        /* ================================
           METRICS
        ================================= */

        [data-testid="stMetric"] {
            background:
                linear-gradient(
                    145deg,
                    #111920,
                    #0C1218
                );

            border: 1px solid #25323D;

            border-radius: 9px;

            padding: 15px 17px;

            min-height: 105px;
        }

        [data-testid="stMetricLabel"] {
            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 9px;

            letter-spacing: 1px;

            color: #71808C;
        }

        [data-testid="stMetricValue"] {
            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 27px;

            font-weight: 600;

            color: #E7EDF3;
        }


        /* ================================
           LIVE VIDEO
        ================================= */

        [data-testid="stImage"] {
            background: #080C10;

            border: 1px solid #25323D;

            border-radius: 9px;

            padding: 5px;

            box-shadow:
                inset 0 0 25px rgba(0,0,0,0.35);
        }


        /* ================================
           DATAFRAME
        ================================= */

        [data-testid="stDataFrame"] {
            border: 1px solid #25323D;

            border-radius: 9px;

            overflow: hidden;
        }


        /* ================================
           STATUS
        ================================= */

        .status-card {
            background: #0D1319;

            border: 1px solid #25323D;

            border-radius: 7px;

            padding: 10px 13px;

            margin-top: 8px;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 9px;

            color: #687782;

            letter-spacing: 0.6px;
        }


        /* ================================
           LIVE BADGE
        ================================= */

        .live-label {
            color: #FF6969;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 9px;

            font-weight: 600;

            letter-spacing: 1px;

            margin-bottom: 8px;
        }


        /* ================================
           FOOTER
        ================================= */

        .footer {
            margin-top: 50px;

            padding-top: 18px;

            border-top: 1px solid #202B35;

            text-align: center;

            color: #46535E;

            font-family:
                "JetBrains Mono",
                monospace;

            font-size: 8px;

            letter-spacing: 1px;
        }


        /* ================================
           ALERTS
        ================================= */

        .stAlert {
            border-radius: 8px !important;
        }


        /* ================================
           PROGRESS
        ================================= */

        [data-testid="stProgress"] {
            margin-top: 8px;
        }


        /* ================================
           MOBILE
        ================================= */

        @media(max-width: 800px) {

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero-title {
                font-size: 20px;
            }

        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

def render_header():

    left, right = st.columns([4.5, 1])

    with left:

        st.markdown(
            """
            <div class="hero-container">

                <div class="hero-title">
                    🚦 AI TRAFFIC INTELLIGENCE
                </div>

                <div class="hero-subtitle">
                    Real-time vehicle detection, tracking
                    and license plate recognition
                </div>

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
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="hero-container">

                <div class="online-badge">
                    ● SYSTEM ONLINE
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SECTION
# ============================================================

def section(title, subtitle=""):

    st.markdown(
        f"""
        <div class="section-title">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )

    if subtitle:

        st.markdown(
            f"""
            <div class="section-subtitle">
                {subtitle}
            </div>
            """,
            unsafe_allow_html=True
        )


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
# PLATE TABLE
# ============================================================

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

    df["_sort"] = (
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
        .sort_values("_sort")
        .drop(columns="_sort")
    )

    return df


# ============================================================
# START
# ============================================================

load_css()

render_header()


# ============================================================
# INPUT FEED
# ============================================================

section(
    "Input Feed",
    "VIDEO ANALYSIS PIPELINE"
)


st.markdown(
    """
    <div class="upload-card">

        <div class="upload-heading">
            TRAFFIC VIDEO SOURCE
        </div>

        <div class="upload-text">
            Upload MP4, AVI, MOV or MKV footage for
            AI-powered traffic analysis.
            Maximum file size: 200 MB.
            <br><br>
            200 MB PER FILE · MP4 · AVI · MOV · MKV
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


uploaded = st.file_uploader(
    "Traffic video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ],
    label_visibility="collapsed"
)


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded is None:

    st.info(
        "🎥  AWAITING VIDEO INPUT  •  Upload traffic footage to initialize the AI detection pipeline."
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

st.success(
    f"FILE READY  •  {uploaded.name}  •  {file_size_mb:.1f} MB"
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


with open(
    video_path,
    "wb"
) as f:

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
    "Loading YOLO11 + ByteTrack + License Plate OCR..."
):

    (
        vehicle_model,
        plate_model,
        reader
    ) = load_models()


st.success(
    "AI ENGINE READY"
)


# ============================================================
# VIDEO
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

section(
    "Live Feed",
    "REAL-TIME COMPUTER VISION INFERENCE"
)


st.markdown(
    """
    <div class="live-label">
        ● LIVE INFERENCE
    </div>
    """,
    unsafe_allow_html=True
)


video_display = st.empty()

progress = st.progress(0)

status_box = st.empty()


# ============================================================
# LIVE STATISTICS
# ============================================================

section(
    "Live Traffic Statistics",
    "CURRENT FRAME"
)


c1, c2, c3, c4, c5 = st.columns(5)


tracked_metric = c1.empty()

cars_metric = c2.empty()

motorcycles_metric = c3.empty()

buses_metric = c4.empty()

trucks_metric = c5.empty()


# ============================================================
# LICENSE PLATES
# ============================================================

section(
    "License Plates",
    "LIVE OCR DETECTIONS"
)


plate_display = st.empty()


# ============================================================
# PROCESS VIDEO
# ============================================================

start_time = time.time()


while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame_number += 1


    if frame_number % FRAME_SKIP != 0:

        continue


    processed_frames += 1


    # ========================================================
    # VEHICLE TRACKING
    # ========================================================

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


    # ========================================================
    # DETECTIONS
    # ========================================================

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = result.boxes


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
            # VEHICLE TYPE
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
            # COORDINATES
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
                or y2 <= y1
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
                                or py2 <= py1
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
    # FRAME OVERLAY
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

    tracked_metric.metric(
        "CURRENTLY TRACKED",
        len(current_ids)
    )


    cars_metric.metric(
        "CARS",
        cars
    )


    motorcycles_metric.metric(
        "MOTORCYCLES",
        motorcycles
    )


    buses_metric.metric(
        "BUSES",
        buses
    )


    trucks_metric.metric(
        "TRUCKS",
        trucks
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
                        "VEHICLE ID"
                    ),

                "License Plate":
                    st.column_config.TextColumn(
                        "LICENSE PLATE"
                    ),

                "Confidence":
                    st.column_config.NumberColumn(
                        "CONFIDENCE",
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

    elapsed = time.time() - start_time


    processing_fps = (
        processed_frames / elapsed
        if elapsed > 0
        else 0
    )


    status_box.markdown(
        f"""
        <div class="status-card">

        ● PROCESSING
        &nbsp;&nbsp;|&nbsp;&nbsp;
        FRAME {frame_number}/{total_frames}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        TRACK IDS {len(all_ids)}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        PLATES {len(plate_data)}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        PROCESS FPS {processing_fps:.1f}

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
    "🎉 AI Traffic Analysis Completed Successfully"
)


# ============================================================
# FINAL RESULTS
# ============================================================

section(
    "Final Results",
    "ANALYSIS SUMMARY"
)


r1, r2, r3, r4 = st.columns(4)


r1.metric(
    "TRACK IDS OBSERVED",
    len(all_ids)
)


r2.metric(
    "LICENSE PLATES",
    len(plate_data)
)


r3.metric(
    "FRAMES PROCESSED",
    processed_frames
)


r4.metric(
    "VIDEO FPS",
    f"{fps:.0f}"
)


# ============================================================
# FINAL PLATES
# ============================================================

section(
    "Final License Plate Results",
    "OCR CONFIRMED DETECTIONS"
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
                    "VEHICLE ID"
                ),

            "License Plate":
                st.column_config.TextColumn(
                    "LICENSE PLATE"
                ),

            "Confidence":
                st.column_config.NumberColumn(
                    "CONFIDENCE",
                    format="%.2f"
                )
        }
    )

else:

    st.info(
        "No license plates were recognized in this video."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        AI TRAFFIC INTELLIGENCE
        &nbsp;•&nbsp;
        COMPUTER VISION ANALYTICS
        &nbsp;•&nbsp;
        YOLO11
        &nbsp;•&nbsp;
        BYTETRACK
        &nbsp;•&nbsp;
        EASYOCR

    </div>
    """,
    unsafe_allow_html=True
)

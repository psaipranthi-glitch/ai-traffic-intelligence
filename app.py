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

    st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0A0E13;
        --panel: #10161D;
        --panel2: #121A22;
        --border: #26323D;
        --text: #E7EDF3;
        --muted: #8B9AA8;
        --cyan: #00E6C3;
        --amber: #FFB020;
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(0, 230, 195, 0.055),
                transparent 35%
            ),
            var(--bg);
        color: var(--text);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2.8rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       HEADER
       ===================================================== */

    .console-header {
        display: flex;
        align-items: center;
        justify-content: space-between;

        width: 100%;

        padding: 20px 24px;
        margin-bottom: 26px;

        background: var(--panel);

        border: 1px solid var(--border);
        border-radius: 8px;
    }

    .console-title {
        margin: 0;

        font-family: 'JetBrains Mono', monospace;

        font-size: 23px;
        font-weight: 700;

        letter-spacing: 0.6px;

        color: var(--text);
    }

    .console-sub {
        margin: 6px 0 0 0;

        font-size: 13px;

        letter-spacing: 0.4px;

        color: var(--muted);
    }


    /* =====================================================
       LIVE BADGE
       ===================================================== */

    .live-badge {
        display: flex;
        align-items: center;

        gap: 8px;

        padding: 7px 14px;

        border: 1px solid rgba(0, 230, 195, 0.35);

        border-radius: 999px;

        background: rgba(0, 230, 195, 0.06);

        color: var(--cyan);

        font-family: 'JetBrains Mono', monospace;

        font-size: 11px;

        letter-spacing: 1.2px;

        white-space: nowrap;
    }

    .live-dot {
        width: 8px;
        height: 8px;

        border-radius: 50%;

        background: var(--cyan);

        box-shadow: 0 0 8px rgba(0, 230, 195, 0.7);
    }


    /* =====================================================
       SECTION LABEL
       ===================================================== */

    .section-label {
        margin: 24px 0 12px 0;

        padding-left: 10px;

        border-left: 2px solid var(--cyan);

        font-family: 'JetBrains Mono', monospace;

        font-size: 11px;

        font-weight: 600;

        letter-spacing: 1.8px;

        text-transform: uppercase;

        color: var(--muted);
    }


    /* =====================================================
       UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background: var(--panel);

        border: 1px solid var(--border);

        border-radius: 8px;

        padding: 12px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }


    /* =====================================================
       KPI CARDS
       ===================================================== */

    .kpi {
        min-height: 88px;

        padding: 15px 17px;

        background: rgba(16, 22, 29, 0.85);

        border: 1px solid var(--border);

        border-radius: 7px;
    }

    .kpi-label {
        margin-bottom: 7px;

        font-family: 'JetBrains Mono', monospace;

        font-size: 10px;

        font-weight: 500;

        letter-spacing: 1.3px;

        text-transform: uppercase;

        color: var(--muted);
    }

    .kpi-value {
        font-family: 'JetBrains Mono', monospace;

        font-size: 27px;

        font-weight: 600;

        color: var(--text);
    }

    .kpi-value.accent {
        color: var(--cyan);
    }


    /* =====================================================
       VIDEO
       ===================================================== */

    [data-testid="stImage"] {
        width: 100%;

        padding: 7px;

        background: var(--panel);

        border: 1px solid var(--border);

        border-radius: 8px;
    }


    /* =====================================================
       BUTTON
       ===================================================== */

    .stButton > button {
        width: 100%;

        margin-top: 10px;

        padding: 10px 16px;

        border: 1px solid var(--cyan);

        border-radius: 7px;

        background: rgba(0, 230, 195, 0.07);

        color: var(--cyan);

        font-family: 'JetBrains Mono', monospace;

        font-weight: 600;

        letter-spacing: 0.4px;
    }

    .stButton > button:hover {
        background: rgba(0, 230, 195, 0.15);

        border-color: var(--cyan);

        color: var(--cyan);
    }


    /* =====================================================
       DATAFRAME
       ===================================================== */

    [data-testid="stDataFrame"] {
        margin-top: 8px;

        border: 1px solid var(--border);

        border-radius: 7px;

        overflow: hidden;
    }


    /* =====================================================
       ALERTS
       ===================================================== */

    [data-testid="stAlert"] {
        border-radius: 7px;
    }


    /* =====================================================
       STATUS
       ===================================================== */

    .status-line {
        margin-top: 7px;

        font-family: 'JetBrains Mono', monospace;

        font-size: 11px;

        letter-spacing: 0.3px;

        color: var(--muted);
    }


    /* =====================================================
       EMPTY STATE
       ===================================================== */

    .empty-state {
        padding: 22px;

        margin-top: 4px;

        text-align: center;

        border: 1px dashed var(--border);

        border-radius: 7px;

        background: rgba(16, 22, 29, 0.45);

        font-family: 'JetBrains Mono', monospace;

        font-size: 11px;

        letter-spacing: 0.8px;

        color: var(--muted);
    }

    </style>
    """, unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

def render_header():

    st.markdown("""
    <div class="console-header">

        <div>

            <p class="console-title">
                🚦 AI TRAFFIC INTELLIGENCE
            </p>

            <p class="console-sub">
                YOLO11 · ByteTrack · License Plate OCR
            </p>

        </div>

        <div class="live-badge">
            <span class="live-dot"></span>
            SYSTEM READY
        </div>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# KPI
# =========================================================

def kpi_html(label, value, accent=False):

    value_class = "kpi-value accent" if accent else "kpi-value"

    return f"""
    <div class="kpi">

        <div class="kpi-label">
            {label}
        </div>

        <div class="{value_class}">
            {value}
        </div>

    </div>
    """


# =========================================================
# INITIALIZE
# =========================================================

inject_theme()
render_header()


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource(show_spinner=False)
def load_models():

    vehicle_model = YOLO("yolo11n.pt")

    plate_model = YOLO("models/best.pt")

    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False
    )

    return vehicle_model, plate_model, reader


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
# INPUT
# =========================================================

st.markdown(
    '<div class="section-label">Input Feed</div>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed"
)


# =========================================================
# NO VIDEO
# =========================================================

if uploaded is None:

    st.markdown("""
    <div class="empty-state">
        UPLOAD A TRAFFIC VIDEO TO BEGIN ANALYSIS
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# =========================================================
# FILE SIZE
# =========================================================

file_size_mb = uploaded.size / (1024 * 1024)

if file_size_mb > 200:

    st.error(
        "Video is larger than 200 MB. "
        "Please upload a smaller video."
    )

    st.stop()


st.success(
    f"Loaded: {uploaded.name} · {file_size_mb:.1f} MB"
)


# =========================================================
# START
# =========================================================

start = st.button(
    "🚀 START AI TRAFFIC ANALYSIS",
    type="primary"
)


if not start:
    st.stop()


# =========================================================
# TEMP VIDEO FILE
# =========================================================

video_path = "uploaded_traffic.mp4"

with open(video_path, "wb") as f:
    f.write(uploaded.getbuffer())


# =========================================================
# LOAD MODELS
# =========================================================

with st.spinner("Loading AI models..."):

    vehicle_model, plate_model, reader = load_models()


# =========================================================
# VIDEO
# =========================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    st.error("Could not open the uploaded video.")

    st.stop()


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25


# =========================================================
# VARIABLES
# =========================================================

frame_number = 0

processed_frames = 0

all_ids = set()

plate_data = {}

last_ocr = {}

OCR_INTERVAL = 45

FRAME_SKIP = 3


# =========================================================
# LIVE FEED
# =========================================================

st.markdown(
    '<div class="section-label">Live Feed</div>',
    unsafe_allow_html=True
)

video_display = st.empty()

progress = st.progress(0)

status = st.empty()


# =========================================================
# STATISTICS
# =========================================================

st.markdown(
    '<div class="section-label">Live Traffic Statistics</div>',
    unsafe_allow_html=True
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

st.markdown(
    '<div class="section-label">License Plates</div>',
    unsafe_allow_html=True
)

plate_display = st.empty()


# =========================================================
# PROCESS VIDEO
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Process every 3rd frame
    if frame_number % FRAME_SKIP != 0:
        continue

    processed_frames += 1

    # -----------------------------------------------------
    # YOLO TRACKING
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # DETECTIONS
    # -----------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        if boxes.id is not None:

            ids = boxes.id.int().cpu().tolist()

            classes = boxes.cls.int().cpu().tolist()

            coordinates = boxes.xyxy.int().cpu().tolist()


            for vehicle_id, cls, box in zip(
                ids,
                classes,
                coordinates
            ):

                current_ids.add(vehicle_id)
                all_ids.add(vehicle_id)


                # -------------------------------------------------
                # VEHICLE CLASS
                # -------------------------------------------------

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


                # -------------------------------------------------
                # BOUNDING BOX
                # -------------------------------------------------

                x1, y1, x2, y2 = box

                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)


                if x2 <= x1 or y2 <= y1:
                    continue


                crop = frame[
                    y1:y2,
                    x1:x2
                ]


                # -------------------------------------------------
                # OCR
                # -------------------------------------------------

                should_ocr = (
                    vehicle_id not in last_ocr
                    or
                    frame_number - last_ocr[vehicle_id]
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

                                px2 = min(crop.shape[1], px2)
                                py2 = min(crop.shape[0], py2)


                                if px2 <= px1 or py2 <= py1:
                                    continue


                                plate_crop = crop[
                                    py1:py2,
                                    px1:px2
                                ]


                                if plate_crop.size == 0:
                                    continue


                                # -------------------------------------------------
                                # OCR
                                # -------------------------------------------------

                                ocr_results = reader.readtext(
                                    plate_crop,
                                    detail=1
                                )


                                best_text = ""
                                best_conf = 0.0


                                for item in ocr_results:

                                    text = clean_text(item[1])

                                    confidence = float(item[2])


                                    if (
                                        len(text) >= 3
                                        and
                                        confidence > best_conf
                                    ):

                                        best_text = text
                                        best_conf = confidence


                                # -------------------------------------------------
                                # SAVE
                                # -------------------------------------------------

                                if best_text:

                                    old = plate_data.get(
                                        vehicle_id
                                    )


                                    if (
                                        old is None
                                        or
                                        best_conf > old["confidence"]
                                    ):

                                        plate_data[vehicle_id] = {
                                            "text": best_text,
                                            "confidence": best_conf
                                        }


                    except Exception:
                        pass


                # -------------------------------------------------
                # DRAW VEHICLE
                # -------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (195, 230, 0),
                    2
                )


                label = (
                    f"{vehicle_type} "
                    f"ID:{vehicle_id}"
                )


                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(25, y1 - 8)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (195, 230, 0),
                    2
                )


                # -------------------------------------------------
                # PLATE LABEL
                # -------------------------------------------------

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


    # =====================================================
    # FRAME OVERLAY
    # =====================================================

    cv2.putText(
        frame,
        f"Currently Tracked: {len(current_ids)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (195, 230, 0),
        2
    )


    # =====================================================
    # DISPLAY
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
    # KPI
    # =====================================================

    tracked_box.markdown(
        kpi_html(
            "Currently Tracked",
            len(current_ids),
            True
        ),
        unsafe_allow_html=True
    )

    cars_box.markdown(
        kpi_html("Cars", cars),
        unsafe_allow_html=True
    )

    bikes_box.markdown(
        kpi_html("Motorcycles", motorcycles),
        unsafe_allow_html=True
    )

    buses_box.markdown(
        kpi_html("Buses", buses),
        unsafe_allow_html=True
    )

    trucks_box.markdown(
        kpi_html("Trucks", trucks),
        unsafe_allow_html=True
    )


    # =====================================================
    # PLATE TABLE
    # =====================================================

    if plate_data:

        rows = []

        for vehicle_id, data in plate_data.items():

            rows.append({
                "Vehicle ID": f"#{vehicle_id}",
                "License Plate": data["text"],
                "Confidence": round(
                    data["confidence"],
                    2
                )
            })


        plate_df = pd.DataFrame(rows)


        plate_df["SortID"] = (
            plate_df["Vehicle ID"]
            .str.replace("#", "", regex=False)
            .astype(int)
        )


        plate_df = (
            plate_df
            .sort_values("SortID")
            .drop(columns=["SortID"])
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
                frame_number / total_frames,
                1.0
            )
        )


    # =====================================================
    # STATUS
    # =====================================================

    status.markdown(
        f"""
        <div class="status-line">
            FRAME {frame_number}/{total_frames}
            ·
            TRACK IDS OBSERVED: {len(all_ids)}
            ·
            PLATES RECOGNIZED: {len(plate_data)}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# CLEANUP
# =========================================================

cap.release()

progress.progress(1.0)


# =========================================================
# COMPLETED
# =========================================================

st.success(
    "🎉 AI Traffic Analysis Completed!"
)


st.markdown(
    '<div class="section-label">Final Results</div>',
    unsafe_allow_html=True
)


a, b, c = st.columns(3)


a.markdown(
    kpi_html(
        "Track IDs Observed",
        len(all_ids),
        True
    ),
    unsafe_allow_html=True
)


b.markdown(
    kpi_html(
        "License Plates Recognized",
        len(plate_data)
    ),
    unsafe_allow_html=True
)


c.markdown(
    kpi_html(
        "Frames Processed",
        processed_frames
    ),
    unsafe_allow_html=True
)


# =========================================================
# FINAL TABLE
# =========================================================

st.markdown(
    '<div class="section-label">Final License Plate Results</div>',
    unsafe_allow_html=True
)


if plate_data:

    final_rows = []


    for vehicle_id, data in plate_data.items():

        final_rows.append({
            "Vehicle ID": f"#{vehicle_id}",
            "License Plate": data["text"],
            "Confidence": round(
                data["confidence"],
                2
            )
        })


    final_df = pd.DataFrame(final_rows)


    final_df["SortID"] = (
        final_df["Vehicle ID"]
        .str.replace("#", "", regex=False)
        .astype(int)
    )


    final_df = (
        final_df
        .sort_values("SortID")
        .drop(columns=["SortID"])
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

import os
import gc
import re
import tempfile

import cv2
import numpy as np
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
)


# ============================================================
# SETTINGS
# ============================================================

MAX_UPLOAD_MB = 200

# Process only 1 out of every 5 frames
FRAME_SKIP = 5

# Resize large videos before inference
MAX_WIDTH = 960

# OCR is attempted only once per vehicle initially,
# then again after this many processed frames.
OCR_INTERVAL = 75

# Maximum number of new vehicles sent to plate/OCR per frame
MAX_PLATE_CHECKS = 2


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 3.2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    .main-title {
        font-family: monospace;
        font-size: 30px;
        font-weight: 900;
        letter-spacing: 1.5px;
        color: #F5F8FA;
        margin-top: 8px;
        margin-bottom: 5px;
    }

    .main-title span {
        color: #00E6C3;
    }

    .main-sub {
        font-family: monospace;
        font-size: 13px;
        letter-spacing: 1px;
        color: #8D9AA5;
        margin-bottom: 14px;
    }

    .system-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 13px;
        border-radius: 20px;
        border: 1px solid rgba(0,230,195,.40);
        background: rgba(0,230,195,.06);
        color: #00E6C3;
        font-family: monospace;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 22px;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00E6C3;
        box-shadow: 0 0 8px rgba(0,230,195,.8);
    }

    .section-title {
        font-family: monospace;
        color: #00E6C3;
        font-size: 13px;
        font-weight: 900;
        letter-spacing: 1.7px;
        margin-top: 24px;
        margin-bottom: 12px;
        padding-left: 10px;
        border-left: 3px solid #00E6C3;
    }

    .info-box {
        background: rgba(16,22,29,.45);
        border: 1px solid rgba(0,230,195,.16);
        border-radius: 10px;
        padding: 11px 14px;
        color: #AAB6C0;
        font-family: monospace;
        font-size: 12px;
    }

    [data-testid="stFileUploader"] {
        background: rgba(16,22,29,.40) !important;
        border: 1px solid rgba(0,230,195,.22) !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(10,14,19,.55) !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        min-height: 45px;
        border-radius: 9px;
        border: 1px solid #00E6C3;
        background: rgba(0,230,195,.06);
        color: #00E6C3;
        font-family: monospace;
        font-weight: 900;
        letter-spacing: .7px;
    }

    .stButton > button:hover {
        background: rgba(0,230,195,.14);
        color: white;
    }

    [data-testid="stMetric"] {
        background: rgba(16,22,29,.48);
        border: 1px solid rgba(0,230,195,.16);
        border-radius: 12px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #9CAAB5 !important;
        font-family: monospace !important;
    }

    [data-testid="stMetricValue"] {
        color: #F5F8FA !important;
        font-family: monospace !important;
    }

    .plate-container {
        width: 100%;
        border: 1px solid rgba(0,230,195,.24);
        border-radius: 12px;
        overflow: hidden;
        background: rgba(10,14,19,.30);
        margin-top: 8px;
    }

    .plate-table {
        width: 100%;
        border-collapse: collapse;
        background: transparent !important;
        font-family: monospace;
    }

    .plate-table th {
        background: rgba(0,230,195,.06) !important;
        color: #00E6C3 !important;
        padding: 13px 16px;
        text-align: left;
        font-size: 12px;
        letter-spacing: 1px;
        border-bottom: 1px solid rgba(0,230,195,.20);
    }

    .plate-table td {
        background: transparent !important;
        color: #FFFFFF !important;
        padding: 13px 16px;
        font-size: 14px;
        border-bottom: 1px solid rgba(255,255,255,.06);
    }

    .plate-table tr {
        background: transparent !important;
    }

    .plate-table tr:hover {
        background: rgba(0,230,195,.035) !important;
    }

    .vehicle-id {
        color: #00E6C3 !important;
        font-weight: 900;
    }

    .plate-number {
        color: #FFFFFF !important;
        font-weight: 900;
        letter-spacing: 1.2px;
    }

    .confidence {
        color: #DDE6EC !important;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #66737D;
        font-family: monospace;
        font-size: 11px;
        padding-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🚦 AI TRAFFIC <span>INTELLIGENCE</span>
    </div>

    <div class="main-sub">
        YOLO11 · ByteTrack · License Plate OCR
    </div>

    <div class="system-badge">
        <span class="live-dot"></span>
        SYSTEM READY
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LAZY MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_vehicle_model():
    from ultralytics import YOLO
    return YOLO("yolo11n.pt")


@st.cache_resource(show_spinner=False)
def load_plate_model():
    from ultralytics import YOLO
    return YOLO("models/best.pt")


@st.cache_resource(show_spinner=False)
def load_ocr():
    import easyocr

    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False,
    )


# ============================================================
# HELPERS
# ============================================================

def clean_plate_text(text):
    text = str(text).upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def resize_frame(frame):

    h, w = frame.shape[:2]

    if w <= MAX_WIDTH:
        return frame

    scale = MAX_WIDTH / w

    new_w = MAX_WIDTH
    new_h = int(h * scale)

    return cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def render_plate_table(plates):

    if not plates:
        st.markdown(
            """
            <div class="info-box">
                No license plates recognized yet.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows = ""

    for vehicle_id in sorted(plates):

        item = plates[vehicle_id]

        rows += f"""
        <tr>
            <td>
                <span class="vehicle-id">#{vehicle_id}</span>
            </td>
            <td>
                <span class="plate-number">
                    {item["text"]}
                </span>
            </td>
            <td>
                <span class="confidence">
                    {item["confidence"]:.2f}
                </span>
            </td>
        </tr>
        """

    html = f"""
    <div class="plate-container">

        <table class="plate-table">

            <thead>
                <tr>
                    <th>VEHICLE ID</th>
                    <th>LICENSE PLATE</th>
                    <th>CONFIDENCE</th>
                </tr>
            </thead>

            <tbody>
                {rows}
            </tbody>

        </table>

    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def detect_plate_and_ocr(
    vehicle_crop,
    plate_model,
    reader,
):

    if vehicle_crop is None:
        return None

    if vehicle_crop.size == 0:
        return None

    try:

        plate_results = plate_model(
            vehicle_crop,
            conf=0.45,
            imgsz=320,
            max_det=2,
            device="cpu",
            verbose=False,
        )

    except Exception:
        return None

    best_text = ""
    best_conf = 0.0

    for result in plate_results:

        if result.boxes is None:
            continue

        boxes = (
            result.boxes
            .xyxy
            .cpu()
            .numpy()
        )

        for box in boxes:

            px1, py1, px2, py2 = map(
                int,
                box,
            )

            px1 = max(0, px1)
            py1 = max(0, py1)

            px2 = min(
                vehicle_crop.shape[1],
                px2,
            )

            py2 = min(
                vehicle_crop.shape[0],
                py2,
            )

            plate_crop = vehicle_crop[
                py1:py2,
                px1:px2,
            ]

            if plate_crop.size == 0:
                continue

            plate_crop = cv2.resize(
                plate_crop,
                None,
                fx=1.4,
                fy=1.4,
                interpolation=cv2.INTER_CUBIC,
            )

            try:

                ocr_results = reader.readtext(
                    plate_crop,
                    detail=1,
                    paragraph=False,
                    batch_size=1,
                )

            except Exception:
                continue

            for item in ocr_results:

                if len(item) < 3:
                    continue

                text = clean_plate_text(
                    item[1]
                )

                conf = float(item[2])

                if (
                    len(text) >= 3
                    and conf > best_conf
                ):

                    best_text = text
                    best_conf = conf

    if not best_text:
        return None

    return {
        "text": best_text,
        "confidence": best_conf,
    }


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section-title">INPUT FEED</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Traffic Video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
    max_upload_size=MAX_UPLOAD_MB,
)


if uploaded_file is None:

    st.markdown(
        """
        <div class="info-box">
            🎥 Upload a traffic video to begin analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# FILE CHECK
# ============================================================

file_size_mb = uploaded_file.size / (
    1024 * 1024
)

st.success(
    f"Loaded: {uploaded_file.name}"
)

st.caption(
    f"File size: {file_size_mb:.1f} MB"
)


# ============================================================
# SAVE FILE TO DISK
# ============================================================

if "video_path" not in st.session_state:

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp_file.write(
        uploaded_file.getbuffer()
    )

    temp_file.close()

    st.session_state.video_path = (
        temp_file.name
    )


video_path = st.session_state.video_path


# ============================================================
# START BUTTON
# ============================================================

start = st.button(
    "🚀 START AI TRAFFIC ANALYSIS",
    use_container_width=True,
)


if not start:
    st.stop()


# ============================================================
# LOAD VEHICLE MODEL ONLY
# ============================================================

with st.spinner(
    "Loading YOLO11 vehicle detector..."
):

    vehicle_model = load_vehicle_model()


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    video_path
)

if not cap.isOpened():

    st.error(
        "Unable to open the uploaded video."
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
    fps = 25


# ============================================================
# DATA
# ============================================================

frame_number = 0
processed_frames = 0

observed_ids = set()

plate_data = {}

last_ocr_frame = {}

plate_model = None
reader = None

plate_models_loaded = False


# ============================================================
# UI
# ============================================================

st.markdown(
    '<div class="section-title">LIVE FEED</div>',
    unsafe_allow_html=True,
)

frame_placeholder = st.empty()

progress = st.progress(0)

status_placeholder = st.empty()


st.markdown(
    '<div class="section-title">LIVE TRAFFIC STATISTICS</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)

tracked_metric = m1.empty()
cars_metric = m2.empty()
motorcycles_metric = m3.empty()
buses_metric = m4.empty()
trucks_metric = m5.empty()


st.markdown(
    '<div class="section-title">LICENSE PLATES</div>',
    unsafe_allow_html=True,
)

plate_placeholder = st.empty()


# ============================================================
# PROCESS VIDEO
# ============================================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # SKIP FRAMES
        # ----------------------------------------------------

        if frame_number % FRAME_SKIP != 0:
            continue

        processed_frames += 1


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        frame = resize_frame(frame)


        # ----------------------------------------------------
        # YOLO + BYTETRACK
        # ----------------------------------------------------

        try:

            results = vehicle_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.40,
                iou=0.50,
                imgsz=640,
                device="cpu",
                verbose=False,
            )

        except Exception as e:

            cap.release()

            st.error(
                f"Vehicle tracking failed: {e}"
            )

            st.stop()


        result = results[0]


        current_ids = set()

        cars = 0
        motorcycles = 0
        buses = 0
        trucks = 0

        candidates = []


        # ====================================================
        # VEHICLES
        # ====================================================

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            ids = (
                result.boxes.id
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
            )

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )


            for vehicle_id, cls, box in zip(
                ids,
                classes,
                boxes,
            ):

                vehicle_id = int(
                    vehicle_id
                )

                cls = int(cls)

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


                current_ids.add(
                    vehicle_id
                )

                observed_ids.add(
                    vehicle_id
                )


                x1, y1, x2, y2 = map(
                    int,
                    box,
                )


                x1 = max(
                    0,
                    x1,
                )

                y1 = max(
                    0,
                    y1,
                )

                x2 = min(
                    frame.shape[1],
                    x2,
                )

                y2 = min(
                    frame.shape[0],
                    y2,
                )


                # ------------------------------------------------
                # PLATE/OCR CANDIDATE
                # ------------------------------------------------

                last_check = (
                    last_ocr_frame
                    .get(
                        vehicle_id,
                        -OCR_INTERVAL,
                    )
                )

                if (
                    frame_number
                    - last_check
                    >= OCR_INTERVAL
                    and len(candidates)
                    < MAX_PLATE_CHECKS
                ):

                    candidates.append(
                        (
                            vehicle_id,
                            x1,
                            y1,
                            x2,
                            y2,
                        )
                    )


                # ------------------------------------------------
                # DRAW VEHICLE
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (195, 230, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"{vehicle_type} ID:{vehicle_id}",
                    (
                        x1,
                        max(
                            22,
                            y1 - 7,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (195, 230, 0),
                    2,
                )


                # ------------------------------------------------
                # DRAW KNOWN PLATE
                # ------------------------------------------------

                if vehicle_id in plate_data:

                    plate_text = (
                        plate_data[
                            vehicle_id
                        ]["text"]
                    )

                    cv2.putText(
                        frame,
                        f"Plate: {plate_text}",
                        (
                            x1,
                            min(
                                frame.shape[0] - 10,
                                y2 + 22,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (32, 176, 255),
                        2,
                    )


        # ====================================================
        # LOAD OCR MODELS ONLY WHEN NEEDED
        # ====================================================

        if candidates:

            if not plate_models_loaded:

                with st.spinner(
                    "Loading license plate AI..."
                ):

                    plate_model = (
                        load_plate_model()
                    )

                    reader = load_ocr()

                    plate_models_loaded = True


            # ------------------------------------------------
            # OCR CANDIDATES
            # ------------------------------------------------

            for (
                vehicle_id,
                x1,
                y1,
                x2,
                y2,
            ) in candidates:

                last_ocr_frame[
                    vehicle_id
                ] = frame_number


                vehicle_crop = frame[
                    y1:y2,
                    x1:x2,
                ]


                if vehicle_crop.size == 0:
                    continue


                plate_result = (
                    detect_plate_and_ocr(
                        vehicle_crop,
                        plate_model,
                        reader,
                    )
                )


                if plate_result:

                    old = plate_data.get(
                        vehicle_id
                    )

                    if (
                        old is None
                        or
                        plate_result[
                            "confidence"
                        ]
                        > old[
                            "confidence"
                        ]
                    ):

                        plate_data[
                            vehicle_id
                        ] = plate_result


        # ====================================================
        # OVERLAY
        # ====================================================

        cv2.putText(
            frame,
            f"TRACKED: {len(current_ids)}",
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (195, 230, 0),
            2,
        )


        # ====================================================
        # DISPLAY FRAME
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame_placeholder.image(
            rgb,
            channels="RGB",
            use_container_width=True,
        )


        # ====================================================
        # METRICS
        # ====================================================

        tracked_metric.metric(
            "Currently Tracked",
            len(current_ids),
        )

        cars_metric.metric(
            "Cars",
            cars,
        )

        motorcycles_metric.metric(
            "Motorcycles",
            motorcycles,
        )

        buses_metric.metric(
            "Buses",
            buses,
        )

        trucks_metric.metric(
            "Trucks",
            trucks,
        )


        # ====================================================
        # PLATE TABLE
        # ====================================================

        with plate_placeholder:

            render_plate_table(
                plate_data
            )


        # ====================================================
        # STATUS
        # ====================================================

        percentage = 0

        if total_frames > 0:

            percentage = min(
                frame_number /
                total_frames,
                1.0,
            )

            progress.progress(
                percentage
            )


        status_placeholder.markdown(
            f"""
            <div class="info-box">
                FRAME {frame_number}/{total_frames}
                &nbsp; | &nbsp;
                PROCESSED {processed_frames}
                &nbsp; | &nbsp;
                TRACK IDS OBSERVED {len(observed_ids)}
                &nbsp; | &nbsp;
                PLATES RECOGNIZED {len(plate_data)}
            </div>
            """,
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # MEMORY CLEANUP
        # ----------------------------------------------------

        del frame
        del rgb
        del result
        del results

        if processed_frames % 20 == 0:
            gc.collect()


finally:

    cap.release()

    gc.collect()


# ============================================================
# COMPLETED
# ============================================================

progress.progress(1.0)

st.success(
    "🎉 AI Traffic Analysis Completed!"
)


# ============================================================
# FINAL RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">FINAL RESULTS</div>',
    unsafe_allow_html=True,
)

f1, f2, f3 = st.columns(3)

f1.metric(
    "Track IDs Observed",
    len(observed_ids),
)

f2.metric(
    "License Plates Recognized",
    len(plate_data),
)

f3.metric(
    "Frames Processed",
    processed_frames,
)


st.markdown(
    '<div class="section-title">FINAL LICENSE PLATE RESULTS</div>',
    unsafe_allow_html=True,
)

render_plate_table(
    plate_data
)


st.markdown(
    """
    <div class="footer">
        AI TRAFFIC INTELLIGENCE
        · YOLO11
        · ByteTrack
        · LICENSE PLATE OCR
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CLEANUP TEMP FILE
# ============================================================

try:

    if os.path.exists(video_path):
        os.remove(video_path)

except Exception:
    pass

gc.collect()

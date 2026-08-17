
import os
import gc
import re
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import easyocr


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SETTINGS
# ============================================================

MAX_VIDEO_MB = 100

# Process one frame out of every 5 frames
FRAME_SKIP = 5

# Resize before YOLO inference
MAX_WIDTH = 960

# OCR only periodically for each vehicle
OCR_INTERVAL = 60

# Don't let OCR run on every detected vehicle
MAX_OCR_VEHICLES_PER_FRAME = 2


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0A0E13;
        color: #E7EDF3;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 4rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    header {
        background: transparent !important;
    }

    .main-title {
        font-family: monospace;
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #F5F8FA;
        margin-top: 8px;
        margin-bottom: 5px;
    }

    .accent {
        color: #00E6C3;
    }

    .main-subtitle {
        font-family: monospace;
        font-size: 13px;
        color: #8F9DA9;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }

    .system-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 20px;
        border: 1px solid rgba(0, 230, 195, 0.45);
        background: rgba(0, 230, 195, 0.06);
        color: #00E6C3;
        font-family: monospace;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 15px;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00E6C3;
        box-shadow: 0 0 8px rgba(0, 230, 195, 0.8);
    }

    .section-title {
        font-family: monospace;
        color: #00E6C3;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1.8px;
        margin-top: 28px;
        margin-bottom: 14px;
        padding-left: 11px;
        border-left: 3px solid #00E6C3;
    }

    [data-testid="stFileUploader"] {
        background: rgba(16, 22, 29, 0.55) !important;
        border: 1px solid rgba(0, 230, 195, 0.25) !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(10, 14, 19, 0.65) !important;
        border: 1px dashed rgba(0, 230, 195, 0.30) !important;
        border-radius: 9px !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 46px;
        background: rgba(0, 230, 195, 0.06) !important;
        border: 1px solid #00E6C3 !important;
        border-radius: 9px !important;
        color: #00E6C3 !important;
        font-family: monospace !important;
        font-weight: 800 !important;
        letter-spacing: 0.8px;
    }

    .stButton > button:hover {
        background: rgba(0, 230, 195, 0.15) !important;
        color: #FFFFFF !important;
    }

    [data-testid="stMetric"] {
        background: rgba(16, 22, 29, 0.55) !important;
        border: 1px solid rgba(0, 230, 195, 0.18) !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #9BA9B5 !important;
        font-family: monospace !important;
    }

    [data-testid="stMetricValue"] {
        color: #F5F8FA !important;
        font-family: monospace !important;
        font-weight: 800 !important;
    }

    .plate-wrapper {
        width: 100%;
        margin-top: 8px;
        margin-bottom: 20px;
        border: 1px solid rgba(0, 230, 195, 0.28);
        border-radius: 12px;
        overflow: hidden;
        background: rgba(10, 14, 19, 0.38);
    }

    .plate-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        background: transparent !important;
        font-family: monospace;
    }

    .plate-table th {
        background: rgba(0, 230, 195, 0.07) !important;
        color: #00E6C3 !important;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1px;
        text-align: left;
        padding: 14px 17px;
        border-bottom: 1px solid rgba(0, 230, 195, 0.25);
    }

    .plate-table td {
        background: transparent !important;
        color: #FFFFFF !important;
        font-size: 14px;
        padding: 14px 17px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }

    .plate-table tr {
        background: transparent !important;
    }

    .plate-table tbody tr:hover {
        background: rgba(0, 230, 195, 0.045) !important;
    }

    .plate-table tbody tr:last-child td {
        border-bottom: none;
    }

    .vehicle-id {
        color: #00E6C3 !important;
        font-weight: 800 !important;
    }

    .plate-number {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        letter-spacing: 1.3px;
    }

    .plate-confidence {
        color: #DCE6EC !important;
        font-weight: 700 !important;
    }

    .status-box {
        background: rgba(16, 22, 29, 0.45);
        border: 1px solid rgba(0, 230, 195, 0.16);
        border-radius: 10px;
        padding: 10px 14px;
        color: #9EADB8;
        font-family: monospace;
        font-size: 12px;
    }

    .footer {
        text-align: center;
        color: #64727E;
        font-family: monospace;
        font-size: 11px;
        padding-top: 20px;
        padding-bottom: 10px;
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
        🚦 AI TRAFFIC <span class="accent">INTELLIGENCE</span>
    </div>

    <div class="main-subtitle">
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
# LOAD MODELS ONCE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_vehicle_model():
    return YOLO("yolo11n.pt")


@st.cache_resource(show_spinner=False)
def load_plate_model():
    return YOLO("models/best.pt")


@st.cache_resource(show_spinner=False)
def load_ocr():
    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False,
    )


# ============================================================
# OCR CLEANING
# ============================================================

def clean_text(text):

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text,
    )

    return text


# ============================================================
# FRAME RESIZE
# ============================================================

def resize_frame(frame):

    height, width = frame.shape[:2]

    if width <= MAX_WIDTH:
        return frame

    scale = MAX_WIDTH / width

    new_width = MAX_WIDTH
    new_height = int(height * scale)

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# PLATE TABLE
# ============================================================

def render_plate_table(plate_data):

    if not plate_data:

        st.markdown(
            """
            <div class="status-box">
                No license plates recognized yet.
            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    html = """
    <div class="plate-wrapper">
    <table class="plate-table">

    <thead>
    <tr>
        <th style="width:25%;">VEHICLE ID</th>
        <th style="width:50%;">LICENSE PLATE</th>
        <th style="width:25%;">CONFIDENCE</th>
    </tr>
    </thead>

    <tbody>
    """

    for vehicle_id in sorted(plate_data):

        data = plate_data[vehicle_id]

        html += f"""
        <tr>
            <td>
                <span class="vehicle-id">
                    #{vehicle_id}
                </span>
            </td>

            <td>
                <span class="plate-number">
                    {data["text"]}
                </span>
            </td>

            <td>
                <span class="plate-confidence">
                    {data["confidence"]:.2f}
                </span>
            </td>
        </tr>
        """

    html += """
    </tbody>
    </table>
    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section-title">INPUT FEED</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov", "mkv"],
    max_upload_size=MAX_VIDEO_MB,
    help=f"Maximum recommended size: {MAX_VIDEO_MB} MB",
)


if uploaded is None:

    st.markdown(
        """
        <div class="status-box">
            🎥 Upload a traffic video to begin analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# FILE SIZE CHECK
# ============================================================

file_size_mb = uploaded.size / (
    1024 * 1024
)

if file_size_mb > MAX_VIDEO_MB:

    st.error(
        f"Video is {file_size_mb:.1f} MB. "
        f"Please upload a video below {MAX_VIDEO_MB} MB."
    )

    st.stop()


st.success(
    f"Loaded: {uploaded.name}"
)

st.caption(
    f"File size: {file_size_mb:.1f} MB"
)


# ============================================================
# SAVE UPLOAD TO DISK
# ============================================================

video_path = os.path.join(
    "/tmp",
    "traffic_input.mp4",
)

with open(
    video_path,
    "wb",
) as f:

    f.write(
        uploaded.getbuffer()
    )


# ============================================================
# START
# ============================================================

if not st.button(
    "🚀 START AI TRAFFIC ANALYSIS",
    use_container_width=True,
):

    st.stop()


# ============================================================
# LOAD VEHICLE MODEL
# ============================================================

with st.spinner(
    "Loading YOLO11..."
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

all_ids = set()

plate_data = {}

last_ocr_frame = {}

last_table_count = -1


# ============================================================
# UI
# ============================================================

st.markdown(
    '<div class="section-title">LIVE FEED</div>',
    unsafe_allow_html=True,
)

video_display = st.empty()

progress_bar = st.progress(0)

status_display = st.empty()


st.markdown(
    '<div class="section-title">LIVE TRAFFIC STATISTICS</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)

tracked_metric = c1.empty()
cars_metric = c2.empty()
motorcycles_metric = c3.empty()
buses_metric = c4.empty()
trucks_metric = c5.empty()


st.markdown(
    '<div class="section-title">LICENSE PLATES</div>',
    unsafe_allow_html=True,
)

plate_display = st.empty()


# ============================================================
# VIDEO LOOP
# ============================================================

try:

    while True:

        ret, original_frame = cap.read()

        if not ret:
            break

        frame_number += 1


        # ----------------------------------------------------
        # FRAME SKIP
        # ----------------------------------------------------

        if frame_number % FRAME_SKIP != 0:
            continue


        processed_frames += 1


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        frame = resize_frame(
            original_frame
        )

        del original_frame


        # ----------------------------------------------------
        # VEHICLE DETECTION + BYTETRACK
        # ----------------------------------------------------

        results = vehicle_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.40,
            iou=0.50,
            imgsz=640,
            device="cpu",
            verbose=False,
            stream=False,
        )


        result = results[0]


        current_ids = set()

        cars = 0
        motorcycles = 0
        buses = 0
        trucks = 0


        # ----------------------------------------------------
        # DETECTIONS
        # ----------------------------------------------------

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

            coordinates = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )


            ocr_candidates = []


            for vehicle_id, cls, box in zip(
                ids,
                classes,
                coordinates,
            ):

                vehicle_id = int(
                    vehicle_id
                )

                cls = int(cls)


                # ------------------------------------------------
                # VEHICLE TYPE
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


                current_ids.add(
                    vehicle_id
                )

                all_ids.add(
                    vehicle_id
                )


                # ------------------------------------------------
                # BOX
                # ------------------------------------------------

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
                # OCR CANDIDATE
                # ------------------------------------------------

                if (
                    vehicle_id not in plate_data
                    and
                    (
                        vehicle_id not in last_ocr_frame
                        or
                        frame_number
                        - last_ocr_frame[vehicle_id]
                        >= OCR_INTERVAL
                    )
                    and
                    len(ocr_candidates)
                    < MAX_OCR_VEHICLES_PER_FRAME
                ):

                    ocr_candidates.append(
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
                            y1 - 8,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (195, 230, 0),
                    2,
                )


                # ------------------------------------------------
                # DRAW KNOWN PLATE
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
                                y2 + 22,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (32, 176, 255),
                        2,
                    )


        # ========================================================
        # PLATE MODEL + OCR
        # ========================================================

        if ocr_candidates:

            try:

                plate_model = load_plate_model()

                reader = load_ocr()


                for (
                    vehicle_id,
                    x1,
                    y1,
                    x2,
                    y2,
                ) in ocr_candidates:

                    vehicle_crop = frame[
                        y1:y2,
                        x1:x2,
                    ]


                    if vehicle_crop.size == 0:
                        continue


                    last_ocr_frame[
                        vehicle_id
                    ] = frame_number


                    # ------------------------------------------------
                    # PLATE DETECTION
                    # ------------------------------------------------

                    plate_results = plate_model(
                        vehicle_crop,
                        conf=0.45,
                        imgsz=320,
                        device="cpu",
                        max_det=2,
                        verbose=False,
                    )


                    best_text = ""

                    best_confidence = 0.0


                    for plate_result in plate_results:

                        if plate_result.boxes is None:
                            continue


                        pboxes = (
                            plate_result
                            .boxes
                            .xyxy
                            .cpu()
                            .numpy()
                        )


                        for pbox in pboxes:

                            px1, py1, px2, py2 = map(
                                int,
                                pbox,
                            )


                            px1 = max(
                                0,
                                px1,
                            )

                            py1 = max(
                                0,
                                py1,
                            )

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


                            # ------------------------------------------------
                            # SMALL OCR IMAGE
                            # ------------------------------------------------

                            plate_crop = cv2.resize(
                                plate_crop,
                                None,
                                fx=1.5,
                                fy=1.5,
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

                                ocr_results = []


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
                                    confidence
                                    > best_confidence
                                ):

                                    best_text = text

                                    best_confidence = (
                                        confidence
                                    )


                    # ------------------------------------------------
                    # SAVE BEST RESULT
                    # ------------------------------------------------

                    if best_text:

                        old = plate_data.get(
                            vehicle_id
                        )


                        if (
                            old is None
                            or
                            best_confidence
                            > old["confidence"]
                        ):

                            plate_data[
                                vehicle_id
                            ] = {
                                "text": best_text,
                                "confidence":
                                    best_confidence,
                            }


                    del vehicle_crop


            except Exception as e:

                # OCR failure should not kill vehicle tracking
                pass


        # ========================================================
        # OVERLAY
        # ========================================================

        cv2.putText(
            frame,
            f"TRACKED: {len(current_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (195, 230, 0),
            2,
        )


        # ========================================================
        # DISPLAY
        # ========================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )


        video_display.image(
            rgb_frame,
            channels="RGB",
            use_container_width=True,
        )


        # ========================================================
        # METRICS
        # ========================================================

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


        # ========================================================
        # TABLE
        # Only rerender when plate count changes
        # ========================================================

        if len(plate_data) != last_table_count:

            with plate_display:

                render_plate_table(
                    plate_data
                )

            last_table_count = len(
                plate_data
            )


        # ========================================================
        # PROGRESS
        # ========================================================

        if total_frames > 0:

            progress = (
                frame_number /
                total_frames
            )

            progress_bar.progress(
                min(
                    progress,
                    1.0,
                )
            )


        # ========================================================
        # STATUS
        # ========================================================

        status_display.markdown(
            f"""
            <div class="status-box">

            FRAME {frame_number}/{total_frames}
            &nbsp; | &nbsp;

            PROCESSED: {processed_frames}
            &nbsp; | &nbsp;

            TRACK IDS OBSERVED: {len(all_ids)}
            &nbsp; | &nbsp;

            PLATES RECOGNIZED: {len(plate_data)}

            </div>
            """,
            unsafe_allow_html=True,
        )


        # --------------------------------------------------------
        # EXPLICIT CLEANUP
        # --------------------------------------------------------

        del result
        del results
        del frame
        del rgb_frame


        # Periodic Python cleanup
        if processed_frames % 25 == 0:

            gc.collect()


finally:

    cap.release()

    gc.collect()


# ============================================================
# FINAL PROGRESS
# ============================================================

progress_bar.progress(
    1.0
)


# ============================================================
# COMPLETION
# ============================================================

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
    len(all_ids),
)

f2.metric(
    "License Plates Recognized",
    len(plate_data),
)

f3.metric(
    "Frames Processed",
    processed_frames,
)


# ============================================================
# FINAL PLATE TABLE
# ============================================================

st.markdown(
    '<div class="section-title">FINAL LICENSE PLATE RESULTS</div>',
    unsafe_allow_html=True,
)


render_plate_table(
    plate_data
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AI TRAFFIC INTELLIGENCE
        · YOLO11
        · ByteTrack
        · License Plate OCR
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CLEANUP
# ============================================================

gc.collect()


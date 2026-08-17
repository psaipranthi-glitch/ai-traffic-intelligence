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
    layout="wide"
)


# =========================================================
# CUSTOM CSS ONLY
# No custom HTML components for header/table
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0A0E13;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 0;
    }

    .subtitle {
        color: #7C8B99;
        font-family: monospace;
        font-size: 14px;
        margin-top: 4px;
    }

    .section-title {
        color: #00E6C3;
        font-family: monospace;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .status-ready {
        color: #00E6C3;
        font-family: monospace;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 8px 14px;
        border: 1px solid #00E6C3;
        border-radius: 20px;
        display: inline-block;
        background: rgba(0, 230, 195, 0.06);
    }

    .stButton > button {
        border: 1px solid #00E6C3;
        color: #00E6C3;
        background: rgba(0, 230, 195, 0.08);
        font-family: monospace;
        font-weight: 700;
        border-radius: 6px;
        min-height: 45px;
    }

    .stButton > button:hover {
        border-color: #00E6C3;
        color: #00E6C3;
        background: rgba(0, 230, 195, 0.18);
    }

    div[data-testid="stProgress"] > div > div > div {
        background-color: #00E6C3 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

title_col, status_col = st.columns(
    [4, 1],
    vertical_alignment="center"
)

with title_col:

    st.markdown(
        '<div class="main-title">🚦 AI TRAFFIC INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'YOLO11 · ByteTrack · License Plate OCR'
        '</div>',
        unsafe_allow_html=True
    )

with status_col:

    st.markdown(
        '<div class="status-ready">● SYSTEM READY</div>',
        unsafe_allow_html=True
    )


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    vehicle_model = YOLO(
        "yolo11n.pt"
    )

    plate_model = YOLO(
        "models/best.pt"
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=False
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

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# =========================================================
# INPUT FEED
# =========================================================

st.markdown(
    '<div class="section-title">Input Feed</div>',
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
    help="Maximum upload size is controlled by Streamlit configuration."
)


# =========================================================
# NO VIDEO
# =========================================================

if uploaded is None:

    st.info(
        "🎥 Upload a traffic video to begin AI analysis."
    )

    st.stop()


# =========================================================
# VIDEO INFORMATION
# =========================================================

file_size_mb = (
    uploaded.size /
    (1024 * 1024)
)

st.success(
    f"Loaded: {uploaded.name}"
)

st.caption(
    f"File size: {file_size_mb:.1f} MB"
)


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
# START BUTTON
# =========================================================

start = st.button(
    "🚀 START AI TRAFFIC ANALYSIS",
    type="primary",
    use_container_width=True
)


# =========================================================
# STOP UNTIL START
# =========================================================

if not start:

    st.warning(
        "Click START AI TRAFFIC ANALYSIS to process the video."
    )

    st.stop()


# =========================================================
# LOAD MODELS
# =========================================================

with st.spinner(
    "Loading YOLO11, ByteTrack and License Plate OCR..."
):

    (
        vehicle_model,
        plate_model,
        reader
    ) = load_models()


# =========================================================
# OPEN VIDEO
# =========================================================

cap = cv2.VideoCapture(
    video_path
)

if not cap.isOpened():

    st.error(
        "Unable to open the uploaded video."
    )

    st.stop()


# =========================================================
# VIDEO INFORMATION
# =========================================================

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


# =========================================================
# TRACKING DATA
# =========================================================

frame_number = 0

all_ids = set()

plate_data = {}

last_ocr = {}

OCR_INTERVAL = 30


# =========================================================
# LIVE FEED
# =========================================================

st.markdown(
    '<div class="section-title">Live Feed</div>',
    unsafe_allow_html=True
)

video_display = st.empty()

progress = st.progress(
    0
)

status = st.empty()


# =========================================================
# LIVE STATISTICS
# =========================================================

st.markdown(
    '<div class="section-title">Live Traffic Statistics</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)

tracked_box = c1.empty()
cars_box = c2.empty()
motorcycles_box = c3.empty()
buses_box = c4.empty()
trucks_box = c5.empty()


# =========================================================
# LICENSE PLATES
# =========================================================

st.markdown(
    '<div class="section-title">License Plates</div>',
    unsafe_allow_html=True
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


    # -----------------------------------------------------
    # PROCESS EVERY 3RD FRAME
    # -----------------------------------------------------

    if frame_number % 3 != 0:
        continue


    # =====================================================
    # YOLO11 + BYTE TRACK
    # =====================================================

    results = vehicle_model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=0.35,

        iou=0.5,

        verbose=False

    )

    result = results[0]


    # =====================================================
    # FRAME COUNTERS
    # =====================================================

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
                .cpu()
                .numpy()
            )

            classes = (
                boxes.cls
                .cpu()
                .numpy()
            )

            coordinates = (
                boxes.xyxy
                .cpu()
                .numpy()
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

                vehicle_id = int(
                    vehicle_id
                )

                cls = int(cls)


                # -----------------------------------------
                # TRACK IDS
                # -----------------------------------------

                current_ids.add(
                    vehicle_id
                )

                all_ids.add(
                    vehicle_id
                )


                # -----------------------------------------
                # VEHICLE CLASS
                # -----------------------------------------

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


                # -----------------------------------------
                # BOUNDING BOX
                # -----------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

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


                crop = frame[
                    y1:y2,
                    x1:x2
                ]


                # =================================================
                # OCR INTERVAL
                # =================================================

                should_ocr = (

                    vehicle_id
                    not in last_ocr

                    or

                    frame_number
                    -
                    last_ocr[vehicle_id]
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


                    # =============================================
                    # LICENSE PLATE DETECTION
                    # =============================================

                    plate_results = plate_model(

                        crop,

                        conf=0.45,

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
                            .cpu()
                            .numpy()

                        )


                        for pbox in plate_boxes:

                            px1, py1, px2, py2 = map(
                                int,
                                pbox
                            )


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


                            plate_crop = crop[
                                py1:py2,
                                px1:px2
                            ]


                            if plate_crop.size == 0:
                                continue


                            # =====================================
                            # EASY OCR
                            # =====================================

                            ocr_results = reader.readtext(
                                plate_crop
                            )


                            if not ocr_results:
                                continue


                            best_text = ""
                            best_conf = 0.0


                            for item in ocr_results:

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


                            # =====================================
                            # SAVE BEST RESULT
                            # =====================================

                            if best_text:

                                if (

                                    vehicle_id
                                    not in plate_data

                                    or

                                    best_conf
                                    >
                                    plate_data[
                                        vehicle_id
                                    ][
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
                # VEHICLE ID
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

                    0.6,

                    (195, 230, 0),

                    2

                )


                # =================================================
                # DISPLAY PLATE ON VEHICLE
                # =================================================

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
                                y2 + 25
                            )
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (32, 176, 255),

                        2

                    )


    # =====================================================
    # TRACKING COUNTER ON VIDEO
    # =====================================================

    cv2.putText(

        frame,

        f"Currently Tracked: {len(current_ids)}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (195, 230, 0),

        2

    )


    # =====================================================
    # DISPLAY VIDEO
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
    # LIVE METRICS
    # =====================================================

    tracked_box.metric(
        "Currently Tracked",
        len(current_ids)
    )

    cars_box.metric(
        "Cars",
        cars
    )

    motorcycles_box.metric(
        "Motorcycles",
        motorcycles
    )

    buses_box.metric(
        "Buses",
        buses
    )

    trucks_box.metric(
        "Trucks",
        trucks
    )


    # =====================================================
    # LIVE LICENSE PLATE TABLE
    # =====================================================

    if plate_data:

        plate_rows = []


        for (
            vehicle_id,
            data
        ) in plate_data.items():

            plate_rows.append({

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
            plate_rows
        )


        # Sort vehicle IDs numerically

        plate_df = plate_df.sort_values(

            by="Vehicle ID",

            key=lambda x:
                x.str.replace(
                    "#",
                    "",
                    regex=False
                ).astype(int)

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

        percentage = (

            frame_number
            /
            total_frames

        )

        progress.progress(

            min(
                percentage,
                1.0
            )

        )


    # =====================================================
    # STATUS
    # =====================================================

    status.info(

        f"FRAME {frame_number}/{total_frames}  |  "
        f"TRACK IDS OBSERVED: {len(all_ids)}  |  "
        f"PLATES RECOGNIZED: {len(plate_data)}"

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

st.markdown(
    '<div class="section-title">Final Results</div>',
    unsafe_allow_html=True
)


f1, f2, f3 = st.columns(3)


f1.metric(
    "Track IDs Observed",
    len(all_ids)
)


f2.metric(
    "License Plates Recognized",
    len(plate_data)
)


f3.metric(
    "Frames Processed",
    frame_number // 3
)


# =========================================================
# FINAL LICENSE PLATE TABLE
# =========================================================

st.markdown(
    '<div class="section-title">Final License Plate Results</div>',
    unsafe_allow_html=True
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


    final_df = final_df.sort_values(

        by="Vehicle ID",

        key=lambda x:
            x.str.replace(
                "#",
                "",
                regex=False
            ).astype(int)

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

st.markdown("---")

st.caption(
    "AI Traffic Intelligence · "
    "YOLO11 + ByteTrack + License Plate OCR"
)
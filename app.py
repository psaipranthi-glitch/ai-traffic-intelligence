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
# SIMPLE DARK UI
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0A0E13;
    }

    .block-container {
        padding-top: 3rem;
        max-width: 1400px;
    }

    h1 {
        font-family: monospace !important;
        letter-spacing: 1px;
    }

    .section {
        font-family: monospace;
        color: #8B9AA8;
        border-left: 3px solid #00E6C3;
        padding-left: 10px;
        margin-top: 25px;
        margin-bottom: 12px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .kpi {
        background: #10161D;
        border: 1px solid #26323D;
        border-radius: 7px;
        padding: 15px;
    }

    .kpi-label {
        color: #8B9AA8;
        font-family: monospace;
        font-size: 11px;
    }

    .kpi-value {
        color: #E7EDF3;
        font-family: monospace;
        font-size: 28px;
        font-weight: 600;
    }

    .accent {
        color: #00E6C3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.title("🚦 AI TRAFFIC INTELLIGENCE")

st.caption(
    "YOLO11 · ByteTrack · License Plate OCR"
)

st.success("● SYSTEM READY")


# =========================================================
# MODEL LOADING
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

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )


# =========================================================
# KPI
# =========================================================

def show_kpi(container, label, value, accent=False):

    color_class = "accent" if accent else ""

    container.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">
                {label}
            </div>
            <div class="kpi-value {color_class}">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# INPUT
# =========================================================

st.markdown(
    '<div class="section">Input Feed</div>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov", "mkv"]
)


if uploaded is None:

    st.info(
        "Upload a traffic video to begin analysis."
    )

    st.stop()


# =========================================================
# FILE SIZE
# =========================================================

file_size_mb = uploaded.size / (1024 * 1024)

if file_size_mb > 200:

    st.error(
        f"File is {file_size_mb:.1f} MB. "
        "Please upload a video below 200 MB."
    )

    st.stop()


st.success(
    f"Loaded: {uploaded.name} · {file_size_mb:.1f} MB"
)


# =========================================================
# START
# =========================================================

start = st.button(
    "🚀 Start AI Traffic Analysis",
    type="primary",
    use_container_width=True
)


if not start:

    st.stop()


# =========================================================
# SAVE VIDEO
# =========================================================

video_path = "uploaded_traffic.mp4"

with open(video_path, "wb") as f:

    f.write(
        uploaded.getbuffer()
    )


# =========================================================
# LOAD MODELS
# =========================================================

with st.spinner("Loading AI models..."):

    (
        vehicle_model,
        plate_model,
        reader
    ) = load_models()


# =========================================================
# VIDEO
# =========================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    st.error("Could not open video.")

    st.stop()


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


# =========================================================
# VARIABLES
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

st.markdown(
    '<div class="section">Live Feed</div>',
    unsafe_allow_html=True
)

video_display = st.empty()

progress = st.progress(0)

status = st.empty()


# =========================================================
# STATISTICS
# =========================================================

st.markdown(
    '<div class="section">Live Traffic Statistics</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)

tracked_box = c1.empty()
cars_box = c2.empty()
bikes_box = c3.empty()
buses_box = c4.empty()
trucks_box = c5.empty()


# =========================================================
# PLATES
# =========================================================

st.markdown(
    '<div class="section">License Plates</div>',
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

    if frame_number % FRAME_SKIP != 0:
        continue

    processed_frames += 1

    # -----------------------------------------------------
    # TRACK VEHICLES
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


                # -------------------------------------------------
                # VEHICLE TYPE
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
                # BOX
                # -------------------------------------------------

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


                # -------------------------------------------------
                # OCR
                # -------------------------------------------------

                should_ocr = (
                    vehicle_id not in last_ocr
                    or
                    frame_number - last_ocr[vehicle_id]
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
                                        plate_crop
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

                                    old = plate_data.get(
                                        vehicle_id
                                    )


                                    if (
                                        old is None
                                        or
                                        best_conf >
                                        old["confidence"]
                                    ):

                                        plate_data[
                                            vehicle_id
                                        ] = {
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
                # DRAW PLATE
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
    # DISPLAY
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
    # KPIs
    # =====================================================

    show_kpi(
        tracked_box,
        "Currently Tracked",
        len(current_ids),
        True
    )

    show_kpi(
        cars_box,
        "Cars",
        cars
    )

    show_kpi(
        bikes_box,
        "Motorcycles",
        motorcycles
    )

    show_kpi(
        buses_box,
        "Buses",
        buses
    )

    show_kpi(
        trucks_box,
        "Trucks",
        trucks
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
            hide_index=True
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


    status.caption(
        f"FRAME {frame_number}/{total_frames} "
        f"· TRACK IDS OBSERVED: {len(all_ids)} "
        f"· PLATES RECOGNIZED: {len(plate_data)}"
    )


# =========================================================
# FINISH
# =========================================================

cap.release()

progress.progress(1.0)

st.success(
    "🎉 AI Traffic Analysis Completed!"
)


# =========================================================
# FINAL RESULTS
# =========================================================

st.markdown(
    '<div class="section">Final Results</div>',
    unsafe_allow_html=True
)


a, b, c = st.columns(3)


show_kpi(
    a,
    "Track IDs Observed",
    len(all_ids),
    True
)

show_kpi(
    b,
    "License Plates Recognized",
    len(plate_data)
)

show_kpi(
    c,
    "Frames Processed",
    processed_frames
)


# =========================================================
# FINAL PLATE TABLE
# =========================================================

st.markdown(
    '<div class="section">Final License Plate Results</div>',
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
        hide_index=True
    )

else:

    st.info(
        "No license plates were recognized."
    )
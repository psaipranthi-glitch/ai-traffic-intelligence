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
            --bg: #0A0E13;
            --panel: #10161D;
            --border: #26323D;
            --text: #E7EDF3;
            --muted: #8B9AA8;
            --cyan: #00E6C3;
        }

        html,
        body,
        [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 15% 0%,
                    rgba(0,230,195,0.055),
                    transparent 35%
                ),
                #0A0E13;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 3.2rem;
            padding-bottom: 3rem;
        }


        /* =================================================
           HEADER
           ================================================= */

        .main-header {
            background:
                linear-gradient(
                    135deg,
                    rgba(16,22,29,0.98),
                    rgba(10,14,19,0.98)
                );

            border: 1px solid #26323D;

            border-radius: 10px;

            padding: 22px 26px;

            margin-bottom: 28px;

            width: 100%;

            box-sizing: border-box;

            box-shadow:
                0 8px 30px rgba(0,0,0,0.18);
        }

        .header-title {
            display: flex;

            align-items: center;

            gap: 12px;

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 23px;

            font-weight: 700;

            letter-spacing: 0.6px;

            color: #E7EDF3;
        }

        .header-icon {
            color: #00E6C3;

            font-size: 25px;
        }

        .header-subtitle {
            margin-top: 8px;

            font-family:
                'IBM Plex Sans',
                sans-serif;

            font-size: 13px;

            color: #8B9AA8;

            letter-spacing: 0.4px;
        }

        .header-status-row {
            display: flex;

            align-items: center;

            gap: 14px;

            margin-top: 16px;
        }

        .system-status {
            display: inline-flex;

            align-items: center;

            gap: 8px;

            padding: 6px 13px;

            border:
                1px solid
                rgba(0,230,195,0.35);

            border-radius: 20px;

            background:
                rgba(0,230,195,0.06);

            color: #00E6C3;

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 11px;

            font-weight: 600;

            letter-spacing: 1px;
        }

        .status-dot {
            display: inline-block;

            width: 8px;

            height: 8px;

            border-radius: 50%;

            background: #00E6C3;

            box-shadow:
                0 0 8px
                rgba(0,230,195,0.8);
        }

        .engine-status {
            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 10px;

            color: #586774;

            letter-spacing: 0.8px;
        }


        /* =================================================
           SECTION LABEL
           ================================================= */

        .section-label {
            margin: 25px 0 12px 0;

            padding-left: 10px;

            border-left: 2px solid #00E6C3;

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 11px;

            font-weight: 600;

            letter-spacing: 1.8px;

            text-transform: uppercase;

            color: #8B9AA8;
        }


        /* =================================================
           SECTION DESCRIPTION
           ================================================= */

        .section-description {
            margin-top: -3px;

            margin-bottom: 10px;

            font-family:
                'IBM Plex Sans',
                sans-serif;

            font-size: 12px;

            color: #586774;

            letter-spacing: 0.3px;
        }


        /* =================================================
           FILE UPLOADER
           ================================================= */

        [data-testid="stFileUploader"] {
            background:
                rgba(16,22,29,0.90);

            border:
                1px solid
                #26323D;

            border-radius: 8px;

            padding: 12px;

            transition:
                border-color 0.2s ease,
                background 0.2s ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color:
                rgba(0,230,195,0.45);

            background:
                rgba(16,22,29,0.98);
        }

        [data-testid="stFileUploaderDropzone"] {
            background:
                rgba(10,14,19,0.55);

            border-radius: 6px;
        }


        /* =================================================
           UPLOAD TEXT
           ================================================= */

        [data-testid="stFileUploaderDropzone"] div {
            font-family:
                'IBM Plex Sans',
                sans-serif;
        }


        /* =================================================
           KPI
           ================================================= */

        .kpi {
            min-height: 88px;

            padding: 15px 17px;

            background:
                rgba(16,22,29,0.90);

            border:
                1px solid
                #26323D;

            border-radius: 7px;

            box-sizing: border-box;
        }

        .kpi-label {
            margin-bottom: 7px;

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 10px;

            font-weight: 500;

            letter-spacing: 1.3px;

            text-transform: uppercase;

            color: #8B9AA8;
        }

        .kpi-value {
            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 27px;

            font-weight: 600;

            color: #E7EDF3;
        }

        .kpi-value.accent {
            color: #00E6C3;
        }


        /* =================================================
           VIDEO
           ================================================= */

        [data-testid="stImage"] {
            width: 100%;

            padding: 7px;

            background: #10161D;

            border:
                1px solid
                #26323D;

            border-radius: 8px;
        }


        /* =================================================
           BUTTON
           ================================================= */

        .stButton > button {
            width: 100%;

            margin-top: 8px;

            padding: 10px 16px;

            border:
                1px solid
                #00E6C3;

            border-radius: 7px;

            background:
                rgba(0,230,195,0.07);

            color: #00E6C3;

            font-family:
                'JetBrains Mono',
                monospace;

            font-weight: 600;

            letter-spacing: 0.5px;
        }

        .stButton > button:hover {
            background:
                rgba(0,230,195,0.16);

            border-color:
                #00E6C3;

            color:
                #00E6C3;
        }


        /* =================================================
           DATAFRAME
           ================================================= */

        [data-testid="stDataFrame"] {
            border:
                1px solid
                #26323D;

            border-radius: 7px;

            overflow: hidden;

            margin-top: 8px;
        }


        /* =================================================
           EMPTY STATE
           ================================================= */

        .empty-state {
            padding: 22px;

            margin-top: 10px;

            text-align: center;

            background:
                rgba(16,22,29,0.55);

            border:
                1px dashed
                #26323D;

            border-radius: 7px;

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 11px;

            letter-spacing: 0.8px;

            color: #8B9AA8;
        }

        .empty-title {
            font-size: 13px;

            font-weight: 600;

            letter-spacing: 0.8px;

            color: #E7EDF3;

            margin-bottom: 6px;
        }

        .empty-description {
            font-size: 11px;

            color: #8B9AA8;

            letter-spacing: 0.5px;
        }


        /* =================================================
           STATUS
           ================================================= */

        .status-line {
            margin-top: 7px;

            font-family:
                'JetBrains Mono',
                monospace;

            font-size: 11px;

            letter-spacing: 0.3px;

            color: #8B9AA8;
        }


        /* =================================================
           SUCCESS MESSAGE
           ================================================= */

        [data-testid="stAlert"] {
            border-radius: 7px;
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
        <div class="main-header">

            <div class="header-title">

                <span class="header-icon">
                    🚦
                </span>

                AI TRAFFIC INTELLIGENCE

            </div>


            <div class="header-subtitle">

                Real-time traffic monitoring powered by
                YOLO11 · ByteTrack · License Plate OCR

            </div>


            <div class="header-status-row">

                <div class="system-status">

                    <span class="status-dot"></span>

                    SYSTEM ONLINE

                </div>


                <div class="engine-status">

                    MULTIMODAL VISION ENGINE

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# KPI HTML
# =========================================================

def kpi_html(label, value, accent=False):

    value_class = (
        "kpi-value accent"
        if accent
        else
        "kpi-value"
    )

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

st.markdown(
    """
    <div class="section-label">
        Input Feed
    </div>

    <div class="section-description">
        200MB per file &nbsp;•&nbsp; MP4, AVI, MOV, MKV
    </div>
    """,
    unsafe_allow_html=True
)


uploaded = st.file_uploader(
    "Upload a traffic video",
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

            <div class="empty-title">
                READY FOR VIDEO INPUT
            </div>

            <div class="empty-description">
                Upload a traffic video to begin AI analysis
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


st.success(
    f"Loaded: {uploaded.name} · {file_size_mb:.1f} MB"
)


# =========================================================
# START BUTTON
# =========================================================

start = st.button(
    "🚀 START AI TRAFFIC ANALYSIS",
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
# LOAD MODELS
# =========================================================

with st.spinner(
    "Loading AI models..."
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
        "Could not open the uploaded video."
    )

    st.stop()


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
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

st.markdown(
    '<div class="section-label">Live Feed</div>',
    unsafe_allow_html=True
)


video_display = st.empty()

progress = st.progress(0)

status = st.empty()


# =========================================================
# LIVE STATISTICS
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
                # BOX
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


                # =================================================
                # VEHICLE BOX
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
        f"Currently Tracked: {len(current_ids)}",
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
        kpi_html(
            "Cars",
            cars
        ),
        unsafe_allow_html=True
    )


    bikes_box.markdown(
        kpi_html(
            "Motorcycles",
            motorcycles
        ),
        unsafe_allow_html=True
    )


    buses_box.markdown(
        kpi_html(
            "Buses",
            buses
        ),
        unsafe_allow_html=True
    )


    trucks_box.markdown(
        kpi_html(
            "Trucks",
            trucks
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
# FINAL PLATE RESULTS
# =========================================================

st.markdown(
    '<div class="section-label">Final License Plate Results</div>',
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

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
    page_title="AI Traffic Intelligence — Console",
    page_icon="🚦",
    layout="wide"
)


# =========================================================
# DESIGN SYSTEM
# =========================================================

def inject_theme():

    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">

    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
          rel="stylesheet">

    <style>

    :root{
        --bg:#0A0E13;
        --panel:#10161D;
        --panel-alt:#161D26;
        --border:#1E2A35;
        --text:#E7EDF3;
        --text-muted:#7C8B99;
        --cyan:#00E6C3;
        --amber:#FFB020;
        --radius:6px;
    }

    html, body, [class*="css"]{
        font-family:'IBM Plex Sans', sans-serif;
        color:var(--text);
    }

    .stApp{
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(0,230,195,0.06),
                transparent 40%
            ),
            var(--bg);
    }

    header[data-testid="stHeader"]{
        background:transparent;
    }

    .block-container{
        padding-top:1.5rem;
        max-width:1400px;
    }


    /* =====================================================
       HEADER
       ===================================================== */

    .console-header{
        display:flex;
        align-items:center;
        justify-content:space-between;

        border:1px solid var(--border);
        background:var(--panel);

        border-radius:var(--radius);

        padding:18px 24px;
        margin-bottom:22px;
    }

    .console-title{
        font-family:'JetBrains Mono', monospace;
        font-size:22px;
        font-weight:700;
        letter-spacing:0.5px;

        margin:0;
        color:var(--text);
    }

    .console-sub{
        font-size:13px;
        color:var(--text-muted);

        margin-top:4px;
        letter-spacing:0.3px;
    }

    .live-badge{
        display:flex;
        align-items:center;
        gap:8px;

        font-family:'JetBrains Mono', monospace;
        font-size:12px;

        letter-spacing:1px;
        color:var(--cyan);

        border:1px solid rgba(0,230,195,0.35);
        background:rgba(0,230,195,0.06);

        padding:6px 14px;

        border-radius:999px;
    }

    .live-dot{
        width:8px;
        height:8px;

        border-radius:50%;

        background:var(--cyan);

        box-shadow:
            0 0 0 0 rgba(0,230,195,0.6);

        animation:pulse 1.6s infinite;
    }

    @keyframes pulse{

        0%{
            box-shadow:
                0 0 0 0 rgba(0,230,195,0.55);
        }

        70%{
            box-shadow:
                0 0 0 8px rgba(0,230,195,0);
        }

        100%{
            box-shadow:
                0 0 0 0 rgba(0,230,195,0);
        }

    }


    /* =====================================================
       SECTION LABELS
       ===================================================== */

    .section-label{

        font-family:'JetBrains Mono', monospace;

        font-size:12px;

        letter-spacing:2px;

        text-transform:uppercase;

        color:var(--text-muted);

        border-left:2px solid var(--cyan);

        padding-left:10px;

        margin:22px 0 12px 0;
    }


    /* =====================================================
       VIDEO
       ===================================================== */

    [data-testid="stImage"]{

        border:1px solid var(--border);

        background:var(--panel);

        border-radius:var(--radius);

        padding:10px;

        position:relative;
    }


    /* =====================================================
       KPI CARDS
       ===================================================== */

    .kpi{

        border:1px solid var(--border);

        background:var(--panel);

        border-radius:var(--radius);

        padding:14px 16px;

        text-align:left;
    }

    .kpi-label{

        font-family:'JetBrains Mono', monospace;

        font-size:11px;

        letter-spacing:1.5px;

        text-transform:uppercase;

        color:var(--text-muted);

        margin-bottom:6px;
    }

    .kpi-value{

        font-family:'JetBrains Mono', monospace;

        font-size:30px;

        font-weight:600;

        color:var(--text);
    }

    .kpi-value.accent{

        color:var(--cyan);

    }


    /* =====================================================
       STREAMLIT DATAFRAME
       ===================================================== */

    [data-testid="stDataFrame"]{

        border:1px solid var(--border);

        border-radius:var(--radius);

        overflow:hidden;

    }


    /* =====================================================
       STATUS
       ===================================================== */

    .status-line{

        font-family:'JetBrains Mono', monospace;

        font-size:12px;

        color:var(--text-muted);

        margin-top:8px;
    }


    /* =====================================================
       PROGRESS
       ===================================================== */

    div[data-testid="stProgress"] > div > div > div{

        background-color:var(--cyan) !important;

    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button{

        font-family:'JetBrains Mono', monospace;

        letter-spacing:0.5px;

        border-radius:var(--radius);

        border:1px solid var(--cyan);

        background:rgba(0,230,195,0.08);

        color:var(--cyan);
    }

    .stButton > button:hover{

        background:rgba(0,230,195,0.18);

        border-color:var(--cyan);

        color:var(--cyan);
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"]{

        background:var(--panel);

        border:1px solid var(--border);

        border-radius:var(--radius);

        padding:10px;
    }


    /* =====================================================
       SUCCESS MESSAGE
       ===================================================== */

    [data-testid="stAlert"]{

        border-radius:var(--radius);

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
# KPI HTML
# =========================================================

def kpi_html(label, value, accent=False):

    cls = (
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

        <div class="{cls}">
            {value}
        </div>

    </div>
    """


# =========================================================
# THEME
# =========================================================

inject_theme()

render_header()


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
# CLEAN OCR TEXT
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
# VIDEO UPLOAD
# =========================================================

st.markdown(
    '<div class="section-label">Input Feed</div>',
    unsafe_allow_html=True
)


uploaded = st.file_uploader(

    "Upload traffic video",

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
        <div style="
            border:1px dashed #1E2A35;
            border-radius:6px;
            padding:20px;
            text-align:center;
            color:#7C8B99;
            font-family:'JetBrains Mono', monospace;
        ">
            UPLOAD A TRAFFIC VIDEO TO BEGIN ANALYSIS
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# VIDEO UPLOADED
# =========================================================

else:

    # -----------------------------------------------------
    # SAVE VIDEO
    # -----------------------------------------------------

    video_path = "uploaded_traffic.mp4"

    with open(
        video_path,
        "wb"
    ) as f:

        f.write(
            uploaded.getbuffer()
        )


    st.success(
        f"Loaded: {uploaded.name}"
    )


    # -----------------------------------------------------
    # START BUTTON
    # -----------------------------------------------------

    start = st.button(
        "🚀 Start AI Traffic Analysis",
        type="primary"
    )


    # =====================================================
    # START PROCESSING
    # =====================================================

    if start:

        # =================================================
        # LOAD MODELS
        # =================================================

        with st.spinner(
            "Loading AI models..."
        ):

            (
                vehicle_model,
                plate_model,
                reader
            ) = load_models()


        # =================================================
        # OPEN VIDEO
        # =================================================

        cap = cv2.VideoCapture(
            video_path
        )


        if not cap.isOpened():

            st.error(
                "Could not open video."
            )

            st.stop()


        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        # =================================================
        # TRACKING VARIABLES
        # =================================================

        frame_number = 0

        all_ids = set()

        plate_data = {}

        last_ocr = {}


        OCR_INTERVAL = 30


        # =================================================
        # STREAMLIT UI
        # =================================================

        st.markdown(
            '<div class="section-label">Live Feed</div>',
            unsafe_allow_html=True
        )


        video_display = st.empty()

        progress = st.progress(0)

        status = st.empty()


        # =================================================
        # LIVE STATISTICS
        # =================================================

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


        # =================================================
        # LICENSE PLATES
        # =================================================

        st.markdown(
            '<div class="section-label">License Plates</div>',
            unsafe_allow_html=True
        )


        plate_display = st.empty()


        # =================================================
        # PROCESS VIDEO
        # =================================================

        while True:

            ret, frame = cap.read()


            if not ret:

                break


            frame_number += 1


            # -------------------------------------------------
            # PROCESS EVERY 3RD FRAME
            # -------------------------------------------------

            if frame_number % 3 != 0:

                continue


            # =================================================
            # YOLO + BYTE TRACK
            # =================================================

            results = vehicle_model.track(

                frame,

                persist=True,

                tracker="bytetrack.yaml",

                conf=0.35,

                iou=0.5,

                verbose=False

            )


            result = results[0]


            current_ids = set()


            cars = 0

            motorcycles = 0

            buses = 0

            trucks = 0


            # =================================================
            # VEHICLE DETECTIONS
            # =================================================

            if result.boxes is not None:

                boxes = result.boxes


                if boxes.id is not None:

                    ids = boxes.id.cpu().numpy()

                    classes = boxes.cls.cpu().numpy()

                    coordinates = boxes.xyxy.cpu().numpy()


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
                        # BOUNDING BOX
                        # =================================================

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
                            - last_ocr[vehicle_id]
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


                            # =================================================
                            # PLATE DETECTION
                            # =================================================

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


                                    if (
                                        plate_crop.size
                                        == 0
                                    ):

                                        continue


                                    # =================================================
                                    # EASY OCR
                                    # =================================================

                                    ocr = reader.readtext(
                                        plate_crop
                                    )


                                    if not ocr:

                                        continue


                                    best_text = ""

                                    best_conf = 0


                                    for item in ocr:


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


                                    # =================================================
                                    # SAVE BEST OCR RESULT
                                    # =================================================

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

                            0.6,

                            (195, 230, 0),

                            2

                        )


                        # =================================================
                        # PLATE LABEL
                        # =================================================

                        if (
                            vehicle_id
                            in plate_data
                        ):


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


            # =================================================
            # CURRENTLY TRACKED TEXT
            # =================================================

            cv2.putText(

                frame,

                f"Currently Tracked: {len(current_ids)}",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (195, 230, 0),

                2

            )


            # =================================================
            # DISPLAY FRAME
            # =================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            video_display.image(

                rgb,

                channels="RGB",

                use_container_width=True

            )


            # =================================================
            # LIVE KPI CARDS
            # =================================================

            tracked_box.markdown(

                kpi_html(
                    "Currently Tracked",
                    len(current_ids),
                    accent=True
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


            # =================================================
            # FIXED LICENSE PLATE TABLE
            # =================================================

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


                # Sort by vehicle ID
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


            # =================================================
            # PROGRESS
            # =================================================

            if total_frames > 0:

                progress.progress(

                    min(
                        frame_number
                        /
                        total_frames,

                        1.0
                    )

                )


            # =================================================
            # STATUS
            # =================================================

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


        # =====================================================
        # FINISH
        # =====================================================

        cap.release()


        progress.progress(1.0)


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
                accent=True
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
                frame_number // 3
            ),

            unsafe_allow_html=True

        )


        # =====================================================
        # FINAL PLATE TABLE
        # =====================================================

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
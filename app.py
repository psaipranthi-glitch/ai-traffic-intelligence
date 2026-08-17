import os
import gc
import re
import tempfile

import streamlit as st
import cv2


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
        padding-top:2.2rem;
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
       TRANSPARENT LICENSE PLATE TABLE
       ===================================================== */

    .plate-wrapper{

        width:100%;

        border:1px solid var(--border);

        border-radius:var(--radius);

        overflow:hidden;

        background:rgba(16,22,29,0.35);

        margin-top:8px;
    }

    .plate-table{

        width:100%;

        border-collapse:collapse;

        background:transparent !important;

        font-family:'JetBrains Mono', monospace;
    }

    .plate-table th{

        background:rgba(0,230,195,0.05) !important;

        color:var(--cyan) !important;

        font-size:11px;

        letter-spacing:1.3px;

        text-align:left;

        padding:12px 16px;

        border-bottom:1px solid var(--border);
    }

    .plate-table td{

        background:transparent !important;

        color:#FFFFFF !important;

        font-size:13px;

        padding:12px 16px;

        border-bottom:1px solid rgba(255,255,255,0.05);
    }

    .plate-table tr{

        background:transparent !important;
    }

    .plate-table tr:last-child td{

        border-bottom:none;
    }

    .plate-id{

        color:var(--cyan) !important;

        font-weight:600;
    }

    .plate-number{

        color:#FFFFFF !important;

        font-weight:700;

        letter-spacing:1px;
    }

    .plate-confidence{

        color:#DDE6EC !important;
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
# LAZY MODEL LOADING
# =========================================================
# IMPORTANT:
# We do NOT import YOLO or EasyOCR at application startup.
# They are loaded only when required.
# =========================================================

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
        verbose=False
    )


# =========================================================
# CLEAN OCR TEXT
# =========================================================

def clean_text(text):

    text = str(text).upper()

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )


# =========================================================
# PLATE TABLE
# =========================================================

def render_plate_table(plate_data, placeholder=None):

    if not plate_data:

        html = """
        <div style="
            border:1px solid #1E2A35;
            border-radius:6px;
            padding:15px;
            text-align:center;
            color:#7C8B99;
            background:rgba(16,22,29,0.30);
            font-family:'JetBrains Mono',monospace;
            font-size:12px;
        ">
            NO LICENSE PLATES RECOGNIZED YET
        </div>
        """

    else:

        rows = ""

        for vehicle_id, data in sorted(
            plate_data.items(),
            key=lambda item: item[0]
        ):

            rows += f"""
            <tr>

                <td>
                    <span class="plate-id">
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

        html = f"""
        <div class="plate-wrapper">

            <table class="plate-table">

                <thead>

                    <tr>

                        <th>
                            VEHICLE ID
                        </th>

                        <th>
                            LICENSE PLATE
                        </th>

                        <th>
                            CONFIDENCE
                        </th>

                    </tr>

                </thead>

                <tbody>

                    {rows}

                </tbody>

            </table>

        </div>
        """

    if placeholder is not None:

        placeholder.markdown(
            html,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            html,
            unsafe_allow_html=True
        )


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

    st.stop()


# =========================================================
# VIDEO SIZE
# =========================================================

file_size_mb = uploaded.size / (
    1024 * 1024
)

st.success(
    f"Loaded: {uploaded.name}"
)

st.caption(
    f"File size: {file_size_mb:.1f} MB"
)


# =========================================================
# START BUTTON
# =========================================================

start = st.button(
    "🚀 Start AI Traffic Analysis",
    type="primary"
)


# =========================================================
# WAIT FOR START
# =========================================================

if not start:

    st.stop()


# =========================================================
# SAVE VIDEO TO TEMPORARY FILE
# =========================================================

temp_video = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=os.path.splitext(uploaded.name)[1]
)

temp_video.write(
    uploaded.getbuffer()
)

temp_video.close()

video_path = temp_video.name


# =========================================================
# LOAD VEHICLE MODEL ONLY
# =========================================================

with st.spinner(
    "Loading vehicle detection model..."
):

    vehicle_model = load_vehicle_model()


# =========================================================
# OPEN VIDEO
# =========================================================

cap = cv2.VideoCapture(
    video_path
)


if not cap.isOpened():

    os.remove(video_path)

    st.error(
        "Could not open video."
    )

    st.stop()


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


# =========================================================
# TRACKING VARIABLES
# =========================================================

frame_number = 0

processed_frames = 0

all_ids = set()

plate_data = {}

last_ocr = {}


# =========================================================
# PERFORMANCE SETTINGS
# =========================================================

# Process one frame out of every 5.
FRAME_SKIP = 5

# OCR for a vehicle only once every 75 processed frames.
OCR_INTERVAL = 75

# Maximum vehicles sent to plate detector per frame.
MAX_PLATE_CHECKS = 2

# Resize large frames.
MAX_WIDTH = 960


# =========================================================
# LAZY OCR MODELS
# =========================================================

plate_model = None

reader = None

ocr_loaded = False


# =========================================================
# STREAMLIT UI
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

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            break


        frame_number += 1


        # -------------------------------------------------
        # FRAME SKIPPING
        # -------------------------------------------------

        if frame_number % FRAME_SKIP != 0:

            continue


        processed_frames += 1


        # -------------------------------------------------
        # RESIZE FRAME
        # -------------------------------------------------

        height, width = frame.shape[:2]

        if width > MAX_WIDTH:

            scale = MAX_WIDTH / width

            new_width = MAX_WIDTH

            new_height = int(
                height * scale
            )

            frame = cv2.resize(
                frame,
                (
                    new_width,
                    new_height
                ),
                interpolation=cv2.INTER_AREA
            )


        # =================================================
        # YOLO + BYTE TRACK
        # =================================================

        results = vehicle_model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=0.35,

            iou=0.5,

            imgsz=640,

            device="cpu",

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

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = result.boxes

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


            plate_candidates = []


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


                current_ids.add(
                    vehicle_id
                )

                all_ids.add(
                    vehicle_id
                )


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


                # =================================================
                # OCR CANDIDATE
                # =================================================

                previous_ocr = last_ocr.get(
                    vehicle_id,
                    -OCR_INTERVAL
                )


                if (
                    frame_number
                    - previous_ocr
                    >= OCR_INTERVAL
                    and
                    len(plate_candidates)
                    < MAX_PLATE_CHECKS
                ):

                    plate_candidates.append(
                        (
                            vehicle_id,
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )


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

                cv2.putText(

                    frame,

                    f"{vehicle_type} ID:{vehicle_id}",

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
                                y2 + 25
                            )
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (32, 176, 255),

                        2

                    )


            # =================================================
            # LOAD PLATE + OCR ONLY WHEN REQUIRED
            # =================================================

            if plate_candidates:

                if not ocr_loaded:

                    with st.spinner(
                        "Loading license plate AI..."
                    ):

                        plate_model = (
                            load_plate_model()
                        )

                        reader = load_ocr()

                        ocr_loaded = True


                # =================================================
                # PLATE PROCESSING
                # =================================================

                for (
                    vehicle_id,
                    x1,
                    y1,
                    x2,
                    y2
                ) in plate_candidates:


                    last_ocr[
                        vehicle_id
                    ] = frame_number


                    crop = frame[
                        y1:y2,
                        x1:x2
                    ]


                    if crop.size == 0:

                        continue


                    # -------------------------------------------------
                    # LIMIT CROP SIZE
                    # -------------------------------------------------

                    crop_h, crop_w = crop.shape[:2]

                    if crop_w > 640:

                        scale = 640 / crop_w

                        crop = cv2.resize(

                            crop,

                            (
                                640,
                                int(
                                    crop_h * scale
                                )
                            ),

                            interpolation=cv2.INTER_AREA

                        )


                    # =================================================
                    # PLATE DETECTION
                    # =================================================

                    plate_results = plate_model(

                        crop,

                        conf=0.45,

                        imgsz=320,

                        max_det=2,

                        device="cpu",

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


                            # =================================================
                            # EASY OCR
                            # =================================================

                            try:

                                ocr_results = reader.readtext(

                                    plate_crop,

                                    detail=1,

                                    paragraph=False

                                )

                            except Exception:

                                continue


                            if not ocr_results:

                                continue


                            best_text = ""

                            best_conf = 0


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


                            # =================================================
                            # SAVE RESULT
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
        # CURRENTLY TRACKED
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
        # KPI CARDS
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
        # TRANSPARENT PLATE TABLE
        # =================================================

        render_plate_table(
            plate_data,
            plate_display
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


        # =================================================
        # MEMORY CLEANUP
        # =================================================

        del results
        del result
        del rgb

        if processed_frames % 15 == 0:

            gc.collect()


finally:

    cap.release()

    gc.collect()


# =========================================================
# FINISH
# =========================================================

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
        processed_frames
    ),

    unsafe_allow_html=True

)


# =========================================================
# FINAL PLATE TABLE
# =========================================================

st.markdown(
    '<div class="section-label">Final License Plate Results</div>',
    unsafe_allow_html=True
)


render_plate_table(
    plate_data
)


# =========================================================
# CLEANUP TEMP FILE
# =========================================================

try:

    if os.path.exists(video_path):

        os.remove(video_path)

except Exception:

    pass


gc.collect()
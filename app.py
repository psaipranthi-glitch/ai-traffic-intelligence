import streamlit as st
import cv2
import re
from ultralytics import YOLO
import easyocr


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Traffic Intelligence",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Traffic Intelligence")
st.caption("YOLO11 + ByteTrack + License Plate OCR")

st.divider()


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    vehicle_model = YOLO("yolo11n.pt")

    plate_model = YOLO("models/best.pt")

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    return vehicle_model, plate_model, reader


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

uploaded = st.file_uploader(
    "🎥 Upload Traffic Video",
    type=["mp4", "avi", "mov", "mkv"]
)


if uploaded is None:

    st.info(
        "Upload a traffic video to start."
    )

else:

    # -----------------------------------------------------
    # SAVE VIDEO
    # -----------------------------------------------------

    video_path = "uploaded_traffic.mp4"

    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())


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


    if start:

        # =================================================
        # LOAD MODELS
        # =================================================

        with st.spinner(
            "Loading AI models..."
        ):

            vehicle_model, plate_model, reader = (
                load_models()
            )


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


        # OCR every 30 processed frames
        OCR_INTERVAL = 30


        # =================================================
        # STREAMLIT UI
        # =================================================

        video_display = st.empty()

        progress = st.progress(0)

        status = st.empty()


        st.divider()

        st.subheader(
            "📊 Live Traffic Statistics"
        )


        c1, c2, c3, c4, c5 = st.columns(5)


        tracked_box = c1.empty()
        cars_box = c2.empty()
        bikes_box = c3.empty()
        buses_box = c4.empty()
        trucks_box = c5.empty()


        st.divider()

        st.subheader(
            "🔢 License Plates"
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


            # ------------------------------------------------
            # Process every 3rd frame
            # ------------------------------------------------

            if frame_number % 3 != 0:

                continue


            # =================================================
            # YOLO TRACKING
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
            # DETECTIONS
            # =================================================

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


                    # =========================================
                    # EACH VEHICLE
                    # =========================================

                    for vehicle_id, cls, box in zip(
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


                        # =====================================
                        # VEHICLE TYPE
                        # =====================================

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


                        # =====================================
                        # VEHICLE BOX
                        # =====================================

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


                        # =====================================
                        # PLATE DETECTION
                        # =====================================

                        crop = frame[
                            y1:y2,
                            x1:x2
                        ]


                        should_ocr = (
                            vehicle_id not in last_ocr
                            or
                            frame_number
                            -
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


                            plate_results = plate_model(
                                crop,
                                conf=0.45,
                                verbose=False
                            )


                            # =================================
                            # PLATE BOXES
                            # =================================

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


                                    # =============================
                                    # OCR
                                    # =============================

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
                                            confidence
                                            > best_conf
                                        ):

                                            best_text = text

                                            best_conf = (
                                                confidence
                                            )


                                    if best_text:

                                        if (
                                            vehicle_id
                                            not in plate_data
                                            or
                                            best_conf
                                            >
                                            plate_data[
                                                vehicle_id
                                            ]["confidence"]
                                        ):

                                            plate_data[
                                                vehicle_id
                                            ] = {
                                                "text":
                                                    best_text,

                                                "confidence":
                                                    best_conf
                                            }


                        # =====================================
                        # DRAW VEHICLE
                        # =====================================

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )


                        label = (
                            f"{vehicle_type} "
                            f"ID:{vehicle_id}"
                        )


                        cv2.putText(
                            frame,
                            label,
                            (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )


                        # =====================================
                        # SHOW PLATE
                        # =====================================

                        if vehicle_id in plate_data:

                            plate_text = plate_data[
                                vehicle_id
                            ]["text"]


                            cv2.putText(
                                frame,
                                f"Plate: {plate_text}",
                                (x1, min(
                                    frame.shape[0] - 10,
                                    y2 + 25
                                )),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (255, 255, 0),
                                2
                            )


            # =================================================
            # TOP INFO
            # =================================================

            cv2.putText(
                frame,
                f"Currently Tracked: {len(current_ids)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )


            # =================================================
            # DISPLAY
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
            # METRICS
            # =================================================

            tracked_box.metric(
                "Currently Tracked",
                len(current_ids)
            )

            cars_box.metric(
                "Cars",
                cars
            )

            bikes_box.metric(
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


            # =================================================
            # PLATE TABLE
            # =================================================

            if plate_data:

                table = []

                for vehicle_id, data in (
                    plate_data.items()
                ):

                    table.append({
                        "Vehicle ID":
                            vehicle_id,

                        "License Plate":
                            data["text"],

                        "Confidence":
                            f"{data['confidence']:.2f}"
                    })


                plate_display.table(
                    table
                )


            # =================================================
            # PROGRESS
            # =================================================

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_number /
                        total_frames,
                        1.0
                    )
                )


            status.write(
                f"Processing frame "
                f"{frame_number}/{total_frames} | "
                f"Track IDs observed: "
                f"{len(all_ids)} | "
                f"Plates recognized: "
                f"{len(plate_data)}"
            )


        # =====================================================
        # FINISH
        # =====================================================

        cap.release()

        progress.progress(1.0)

        st.success(
            "🎉 AI Traffic Analysis Completed!"
        )


        st.divider()


        st.subheader(
            "📊 Final Results"
        )


        a, b, c = st.columns(3)


        a.metric(
            "Track IDs Observed",
            len(all_ids)
        )


        b.metric(
            "License Plates Recognized",
            len(plate_data)
        )


        c.metric(
            "Frames Processed",
            frame_number // 3
        )
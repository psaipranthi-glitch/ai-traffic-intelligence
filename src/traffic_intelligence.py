import cv2
import easyocr
from ultralytics import YOLO
from collections import defaultdict, Counter
import re
import os

# ==========================================
# CONFIG
# ==========================================

VIDEO_PATH = "data/videos/traffic.mp4"
VEHICLE_MODEL = "yolo11n.pt"
PLATE_MODEL = "models/best.pt"

os.makedirs("outputs", exist_ok=True)

# ==========================================
# LOAD MODELS
# ==========================================

vehicle_model = YOLO(VEHICLE_MODEL)
plate_model = YOLO(PLATE_MODEL)

reader = easyocr.Reader(["en"], gpu=False)

# ==========================================
# VEHICLE CLASSES
# ==========================================

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Store OCR results for each vehicle
plate_history = defaultdict(list)

# ==========================================
# VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

frame_number = 0

print()
print("==============================")
print(" AI TRAFFIC INTELLIGENCE")
print("==============================")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Process every 5th frame
    if frame_number % 5 != 0:
        continue

    # ======================================
    # VEHICLE DETECTION + TRACKING
    # ======================================

    results = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]

    if result.boxes.id is None:
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    ids = result.boxes.id.cpu().numpy().astype(int)

    # ======================================
    # PROCESS VEHICLES
    # ======================================

    for box, cls, vehicle_id in zip(
        boxes,
        classes,
        ids
    ):

        cls = int(cls)

        if cls not in vehicle_classes:
            continue

        vehicle_type = vehicle_classes[cls]

        x1, y1, x2, y2 = map(
            int,
            box
        )

        # ==================================
        # LICENSE PLATE DETECTION
        # ==================================

        vehicle_crop = frame[
            max(0, y1):y2,
            max(0, x1):x2
        ]

        if vehicle_crop.size == 0:
            continue

        plate_results = plate_model(
            vehicle_crop,
            conf=0.25,
            verbose=False
        )

        plate_result = plate_results[0]

        best_plate = None
        best_conf = 0

        # ==================================
        # FIND BEST PLATE
        # ==================================

        if plate_result.boxes is not None:

            for pbox in plate_result.boxes:

                px1, py1, px2, py2 = map(
                    int,
                    pbox.xyxy[0]
                )

                confidence = float(
                    pbox.conf[0]
                )

                if confidence > best_conf:

                    best_conf = confidence

                    best_plate = vehicle_crop[
                        max(0, py1):py2,
                        max(0, px1):px2
                    ]

        # ==================================
        # OCR
        # ==================================

        plate_text = ""

        if (
            best_plate is not None
            and best_plate.size > 0
        ):

            # Upscale
            plate_image = cv2.resize(
                best_plate,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            # Grayscale
            gray = cv2.cvtColor(
                plate_image,
                cv2.COLOR_BGR2GRAY
            )

            # Contrast
            gray = cv2.equalizeHist(gray)

            # OCR
            ocr_results = reader.readtext(
                gray,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            if ocr_results:

                text, confidence = max(
                    [
                        (
                            re.sub(
                                r"[^A-Z0-9]",
                                "",
                                r[1].upper()
                            ),
                            r[2]
                        )
                        for r in ocr_results
                    ],
                    key=lambda x: x[1]
                )

                if (
                    text
                    and confidence >= 0.40
                ):

                    plate_text = text

                    plate_history[
                        vehicle_id
                    ].append(text)

        # ==================================
        # GET MOST COMMON PLATE
        # ==================================

        final_plate = "UNKNOWN"

        if plate_history[vehicle_id]:

            final_plate = Counter(
                plate_history[vehicle_id]
            ).most_common(1)[0][0]

        # ==================================
        # DRAW VEHICLE
        # ==================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = (
            f"ID {vehicle_id} | "
            f"{vehicle_type} | "
            f"{final_plate}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    # ======================================
    # SAVE DISPLAY FRAME
    # ======================================

    cv2.imwrite(
        "outputs/latest_frame.jpg",
        frame
    )

    # Progress
    if frame_number % 100 == 0:

        print(
            f"Processed frames: {frame_number}"
        )

cap.release()

# ==========================================
# FINAL RESULTS
# ==========================================

print()
print("==============================")
print(" TRAFFIC INTELLIGENCE COMPLETE")
print("==============================")

for vehicle_id, plates in plate_history.items():

    if plates:

        final_plate = Counter(
            plates
        ).most_common(1)[0][0]

        print(
            f"Vehicle ID {vehicle_id} → {final_plate}"
        )

print("==============================")
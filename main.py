import cv2
import time
import os
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "data/videos/traffic.mp4"
MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.4

# Low counting line
LINE_Y = 340

# Vehicle classes from COCO
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs("outputs", exist_ok=True)

# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps_video = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_video_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print()
print("Video information")
print("------------------------------")
print("Resolution :", width, "x", height)
print("FPS        :", fps_video)
print("Frames     :", total_video_frames)
print("Line Y     :", LINE_Y)
print("------------------------------")

# ==========================================
# OUTPUT VIDEO
# ==========================================

output_path = "outputs/traffic_counted.mp4"

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps_video,
    (width, height)
)

# ==========================================
# TRACKING DATA
# ==========================================

previous_positions = {}

counted_ids = set()

counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

frame_count = 0

start_time = time.time()

# ==========================================
# PROCESS VIDEO
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_count += 1

    # ======================================
    # YOLO + BYTE TRACK
    # ======================================

    results = model.track(
        frame,
        persist=True,
        classes=list(VEHICLE_CLASSES.keys()),
        conf=CONFIDENCE,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    # ======================================
    # PROCESS DETECTIONS
    # ======================================

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        ids = result.boxes.id.int().cpu().tolist()

        classes = result.boxes.cls.int().cpu().tolist()

        confidences = result.boxes.conf.cpu().tolist()

        for box, track_id, class_id, confidence in zip(
            boxes,
            ids,
            classes,
            confidences
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )

            # ==================================
            # CENTER POINT
            # ==================================

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            vehicle_type = VEHICLE_CLASSES.get(
                class_id,
                "Unknown"
            )

            # ==================================
            # TRACK VEHICLE POSITION
            # ==================================

            previous_y = previous_positions.get(
                track_id
            )

            if previous_y is not None:

                # Moving downward
                crossed_down = (
                    previous_y < LINE_Y
                    and center_y >= LINE_Y
                )

                # Moving upward
                crossed_up = (
                    previous_y > LINE_Y
                    and center_y <= LINE_Y
                )

                # ==================================
                # COUNT VEHICLE
                # ==================================

                if (
                    (crossed_down or crossed_up)
                    and track_id not in counted_ids
                ):

                    counted_ids.add(track_id)

                    if vehicle_type in counts:

                        counts[vehicle_type] += 1

            previous_positions[track_id] = center_y

            # ==================================
            # DRAW BOUNDING BOX
            # ==================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ==================================
            # DRAW CENTER
            # ==================================

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # ==================================
            # LABEL
            # ==================================

            label = (
                f"ID {track_id} | "
                f"{vehicle_type} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # ==========================================
    # DRAW COUNTING LINE
    # ==========================================

    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (255, 0, 0),
        4
    )

    cv2.putText(
        frame,
        "COUNTING LINE",
        (20, LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # ==========================================
    # CALCULATE FPS
    # ==========================================

    elapsed = time.time() - start_time

    fps = (
        frame_count / elapsed
        if elapsed > 0
        else 0
    )

    # ==========================================
    # TOTAL COUNT
    # ==========================================

    total_count = sum(
        counts.values()
    )

    # ==========================================
    # DASHBOARD BACKGROUND
    # ==========================================

    cv2.rectangle(
        frame,
        (10, 10),
        (300, 210),
        (0, 0, 0),
        -1
    )

    # ==========================================
    # DASHBOARD TITLE
    # ==========================================

    cv2.putText(
        frame,
        "AI TRAFFIC INTELLIGENCE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # ==========================================
    # COUNTS
    # ==========================================

    cv2.putText(
        frame,
        f"Cars        : {counts['Car']}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles : {counts['Motorcycle']}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses       : {counts['Bus']}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks      : {counts['Truck']}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"TOTAL       : {total_count}",
        (20, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    # ==========================================
    # FPS DISPLAY
    # ==========================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (width - 130, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # ==========================================
    # SAVE FRAME
    # ==========================================

    out.write(frame)

    # ==========================================
    # SHOW VIDEO
    # ==========================================

    cv2.imshow(
        "AI Traffic Intelligence",
        frame
    )

    # ==========================================
    # PRESS Q TO EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

out.release()

cv2.destroyAllWindows()

# ==========================================
# FINAL RESULTS
# ==========================================

total = sum(
    counts.values()
)

print()
print("=" * 45)
print("       AI TRAFFIC INTELLIGENCE")
print("=" * 45)

print(f"Cars        : {counts['Car']}")
print(f"Motorcycles : {counts['Motorcycle']}")
print(f"Buses       : {counts['Bus']}")
print(f"Trucks      : {counts['Truck']}")

print("-" * 45)

print(f"Total crossed: {total}")

print("=" * 45)

print()
print("Output video:")
print(output_path)

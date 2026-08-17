import cv2
from ultralytics import YOLO
from collections import Counter

VIDEO = "data/videos/traffic.mp4"

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO)

vehicle_ids = set()
vehicle_types = Counter()

frame_count = 0

classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Process every 10th frame
    if frame_count % 10 != 0:
        continue

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]

    if result.boxes.id is None:
        continue

    ids = result.boxes.id.cpu().numpy()
    classes_detected = result.boxes.cls.cpu().numpy()

    for vehicle_id, cls in zip(ids, classes_detected):

        cls = int(cls)

        if cls in classes:

            vehicle_ids.add(int(vehicle_id))
            vehicle_types[classes[cls]] += 1

    if frame_count % 100 == 0:
        print(f"Processed: {frame_count}")

cap.release()

print()
print("==============================")
print(" AI TRAFFIC INTELLIGENCE")
print("==============================")
print(f"Unique vehicles : {len(vehicle_ids)}")
print()
print("Vehicle detections:")
print(f"Cars            : {vehicle_types['Car']}")
print(f"Motorcycles     : {vehicle_types['Motorcycle']}")
print(f"Buses           : {vehicle_types['Bus']}")
print(f"Trucks          : {vehicle_types['Truck']}")
print("==============================")
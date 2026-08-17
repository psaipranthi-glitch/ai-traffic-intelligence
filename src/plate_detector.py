import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "models/best.pt"
IMAGE_PATH = "data/traffic_frame.jpg"

os.makedirs("outputs", exist_ok=True)

model = YOLO(MODEL_PATH)

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found")
    exit()

results = model(
    image,
    conf=0.25,
    verbose=False
)

result = results[0]

plate_count = 0

for box in result.boxes:

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    confidence = float(box.conf[0])

    plate_count += 1

    # Crop plate
    plate = image[y1:y2, x1:x2]

    crop_path = f"outputs/plate_{plate_count}.jpg"

    cv2.imwrite(crop_path, plate)

    # Draw box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Plate {confidence:.2f}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    print(f"Plate {plate_count}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Crop saved: {crop_path}")

cv2.imwrite(
    "outputs/plate_detection.jpg",
    image
)

print()
print("==============================")
print("PLATE DETECTION COMPLETE")
print("==============================")
print("Total plates:", plate_count)
print("==============================")
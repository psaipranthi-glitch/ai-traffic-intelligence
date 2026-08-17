import easyocr
import cv2
import re

reader = easyocr.Reader(['en'], gpu=False)

image = cv2.imread("outputs/plate_1.jpg")

# Upscale
image = cv2.resize(
    image,
    None,
    fx=4,
    fy=4,
    interpolation=cv2.INTER_CUBIC
)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Improve contrast
gray = cv2.equalizeHist(gray)

# Slight sharpening
blur = cv2.GaussianBlur(gray, (3, 3), 0)
sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

# Save processed image
cv2.imwrite(
    "outputs/plate_processed.jpg",
    sharp
)

# OCR
results = reader.readtext(
    sharp,
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

print()
print("==============================")
print("LICENSE PLATE OCR")
print("==============================")

best_text = ""
best_conf = 0

for detection in results:

    text = detection[1].upper()
    confidence = detection[2]

    # Keep only letters/numbers
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    if confidence > best_conf:
        best_text = text
        best_conf = confidence

if best_text:
    print("Plate Text :", best_text)
    print("Confidence :", f"{best_conf:.2f}")
else:
    print("No plate text detected")

print("==============================")
print("Processed image:")
print("outputs/plate_processed.jpg")
print("==============================")
import easyocr

reader = easyocr.Reader(["en"])

image_path = "data/plate.jpg"

results = reader.readtext(image_path)

for detection in results:
    box, text, confidence = detection

    print("Text:", text)
    print("Confidence:", round(confidence, 2))
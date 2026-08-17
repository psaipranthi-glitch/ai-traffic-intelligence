import cv2

video_path = "data/videos/traffic.mp4"

cap = cv2.VideoCapture(video_path)

# Jump to frame 500
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)

success, frame = cap.read()

if success:
    cv2.imwrite("data/traffic_frame.jpg", frame)
    print("Frame saved: data/traffic_frame.jpg")
else:
    print("Could not read frame")

cap.release()
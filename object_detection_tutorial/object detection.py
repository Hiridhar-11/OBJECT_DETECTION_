from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model
model = YOLO(r"object_detection_tutorial.v1i.yolov8\yolov8n.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # --- YOLOv8 Detection ---
    results = model(frame)
    annotated_frame = results[0].plot()

    # --- Canny Edge Detection ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # --- Display Both Images Side-by-Side ---
   # combined = cv2.hconcat([annotated_frame, edges_colored])
    cv2.imshow("YOLOv8 Detection",annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()


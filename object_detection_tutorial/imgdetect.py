from ultralytics import YOLO
import cv2

# 1. Loading your custom-trained model
model = YOLO(r"C:\Users\HIRI\runs\detect\train\weights\best.pt")

# 2. Provide path to your test image
image_file = r"object_detection_tutorial.v1i.yolov8-obb\test\images\LAPTOP3_jpg.rf.f841f7fc7581957e4449f807e6c78079.jpg"

# 3. Perform object detection
results = model(image_file)

# 4. Plot and show results
result_img = results[0].plot()
cv2.imshow("YOLOv8 Detection", result_img)
cv2.waitKey(0)

# 5. (Optionally) Save the result
output_file = r"C:\path\to\your\output.jpg"
cv2.imwrite(output_file, result_img)

# 6. Cleanup
cv2.destroyAllWindows()

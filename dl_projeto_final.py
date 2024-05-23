import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv5 model
model = YOLO('yolo8n.pt')

# Function to perform object detection on a single frame
def detect_objects(frame):
    # Perform object detection on the frame
    results = model.predict(frame)

    # Draw bounding boxes and labels on the frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        label = f'{model.names[box.cls][0]} {box.conf:.2f}'
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

# Open the video file
video_path = 'video_teste2.mp4'
cap = cv2.VideoCapture(video_path)

# Create a VideoWriter object to save the output video
output_path = 'output.mp4'
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    frame_with_detections = detect_objects(frame)

    # Write the frame with detections to the output video
    out.write(frame_with_detections)

    # Display the frame with detections (optional)
    cv2.imshow('frame', frame_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()
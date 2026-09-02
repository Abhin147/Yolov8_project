import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Path to the image you want to detect
image_path = "bus.jpg"

# Read the image
image = cv2.imread(image_path)

if image is None:
    raise RuntimeError("Could not load the image. Check the file path.")

# Run YOLO object detection
results = model(image)

# Draw bounding boxes
for result in results:
    for box in result.boxes:

        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Get confidence score
        conf = box.conf[0].item()

        # Get class ID
        cls = int(box.cls[0].item())

        # Get object name
        label = model.names[cls]

        # Draw bounding box
        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        # Create label text
        text = f"{label} {conf:.2f}"

        # Draw label
        cv2.putText(
            image,
            text,
            (int(x1), max(25, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# Save the resulting image
output_path = "output.jpg"
cv2.imwrite(output_path, image)

print(f"Detection complete!")
print(f"Result saved as: {output_path}")

# Display the resulting image
cv2.imshow("YOLO Detection", image)

# Wait until a key is pressed
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
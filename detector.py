
from ultralytics import YOLO
import cv2

MODEL_PATH = "models/road_damage_best.pt"

# Load trained model
model = YOLO(MODEL_PATH)


def detect_hazards(image_path, confidence=0.25):
    """
    Detect road hazards in an image.
    Returns annotated image, detections, and hazard count.
    """

    results = model.predict(
        source=image_path,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # Draw bounding boxes
    annotated_image = result.plot()

    # Convert BGR image to RGB for display
    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])

            class_name = model.names[class_id]

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            detections.append({
                "hazard": class_name,
                "confidence": round(confidence_score * 100, 2),
                "box": [x1, y1, x2, y2]
            })

    return annotated_image, detections, len(detections)

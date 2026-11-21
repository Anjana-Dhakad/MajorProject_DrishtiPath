
from detector import detect_hazards
from PIL import Image
import os

image_path = "uploads/test_road.jpg"

if not os.path.exists(image_path):
    print("Test image not found!")
else:
    print("Running AI road hazard detection...")

    annotated_image, detections, total = detect_hazards(image_path)

    print("\nDetection completed!")
    print("Total hazards detected:", total)

    for i, detection in enumerate(detections, start=1):
        print(f"\nHazard {i}")
        print("Type:", detection["hazard"])
        print("Confidence:", detection["confidence"], "%")

    output_path = "uploads/detected_output.jpg"

    Image.fromarray(annotated_image).save(output_path)

    print("\nAnnotated image saved at:", output_path)


def calculate_severity(detections, image_width, image_height):

    if not detections:
        return "No Hazards", 0

    total_score = 0

    # Base weights by hazard type
    hazard_weights = {
        "pothole": 3,
        "edge": 3,
        "alligator": 3,
        "block": 2,
        "longitudinal": 2,
        "transverse": 2,
        "crack": 1
    }

    image_area = max(image_width * image_height, 1)

    for detection in detections:

        hazard_type = detection["hazard"].lower()
        confidence = detection["confidence"] / 100

        x1, y1, x2, y2 = detection["box"]

        box_area = max(0, (x2 - x1) * (y2 - y1))
        area_ratio = box_area / image_area

        base_weight = hazard_weights.get(hazard_type, 1)

        # Confidence contribution
        confidence_factor = 0.5 + confidence

        # Relative visual size contribution
        if area_ratio >= 0.20:
            size_factor = 1.5
        elif area_ratio >= 0.08:
            size_factor = 1.25
        elif area_ratio >= 0.02:
            size_factor = 1.0
        else:
            size_factor = 0.75

        detection_score = (
            base_weight
            * confidence_factor
            * size_factor
        )

        total_score += detection_score

    # Preliminary thresholds for prototype
    if total_score >= 12:
        severity = "High"
    elif total_score >= 5:
        severity = "Medium"
    else:
        severity = "Low"

    return severity, round(total_score, 2)

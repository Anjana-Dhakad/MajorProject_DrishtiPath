"""Video-based road hazard detection with cross-frame duplicate suppression."""
from pathlib import Path
import math
import cv2

from detector import model


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(box_a, box_b):
    ax = (box_a[0] + box_a[2]) / 2
    ay = (box_a[1] + box_a[3]) / 2
    bx = (box_b[0] + box_b[2]) / 2
    by = (box_b[1] + box_b[3]) / 2
    return math.hypot(ax - bx, ay - by)


def _assign_track_id(detection, tracks, frame_diagonal, timestamp, max_gap_seconds):
    """Match a detection to a recently seen same-class hazard, otherwise start a track."""
    best_track = None
    best_score = -1.0
    for track in tracks:
        if track["hazard"] != detection["hazard"]:
            continue
        if timestamp - track["last_seen"] > max_gap_seconds:
            continue
        overlap = _iou(detection["box"], track["last_box"])
        distance_ratio = _center_distance(detection["box"], track["last_box"]) / max(frame_diagonal, 1)
        # IoU handles stable views; normalized center distance tolerates modest camera motion.
        if overlap >= 0.15 or distance_ratio <= 0.035:
            score = max(overlap, 1.0 - distance_ratio)
            if score > best_score:
                best_score, best_track = score, track

    if best_track is None:
        new_id = len(tracks) + 1
        best_track = {
            "track_id": new_id,
            "hazard": detection["hazard"],
            "first_seen": timestamp,
            "last_seen": timestamp,
            "last_box": detection["box"],
            "max_confidence": detection["confidence"],
            "observations": 1,
        }
        tracks.append(best_track)
    else:
        best_track["last_seen"] = timestamp
        best_track["last_box"] = detection["box"]
        best_track["max_confidence"] = max(best_track["max_confidence"], detection["confidence"])
        best_track["observations"] += 1

    detection["track_id"] = best_track["track_id"]
    return best_track["track_id"]


def detect_hazards_in_video(video_path, output_path, sample_interval=1.0, confidence=0.25):
    """Analyze sampled frames, associate repeat detections, and save annotated video.

    Returns: output video path, frame-level detections (with track_id), summary.
    Unique hazard counts are based on cross-frame tracks, not raw frame detections.
    """
    video_path, output_path = str(video_path), str(output_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video. Please try another video file.")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total_frames / source_fps if source_fps > 0 else 0
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Could not read the video dimensions.")

    sample_interval = max(float(sample_interval), 0.1)
    frame_step = max(int(round(source_fps * sample_interval)), 1) if source_fps > 0 else 1
    output_fps = max(source_fps / frame_step, 0.1) if source_fps > 0 else 1.0
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the processed video output.")

    detections, tracks = [], []
    analyzed_frames, frame_index = 0, 0
    frame_diagonal = math.hypot(width, height)
    # Allow short gaps between sampled frames, but expire tracks after 3 sample intervals.
    max_gap_seconds = max(sample_interval * 3.1, 2.0)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame_index % frame_step == 0:
                results = model.predict(source=frame, conf=confidence, verbose=False)
                result = results[0]
                annotated = result.plot()
                timestamp = frame_index / source_fps if source_fps > 0 else 0.0

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence_score = float(box.conf[0])
                        class_name = model.names[class_id]
                        coords = list(map(int, box.xyxy[0].tolist()))
                        detection = {
                            "hazard": class_name,
                            "confidence": round(confidence_score * 100, 2),
                            "box": coords,
                            "timestamp_seconds": round(timestamp, 2),
                            "frame_number": frame_index,
                        }
                        track_id = _assign_track_id(detection, tracks, frame_diagonal, timestamp, max_gap_seconds)
                        detections.append(detection)
                        # Add stable track identifier near the YOLO bounding box.
                        x1, y1, _, _ = coords
                        cv2.putText(annotated, f"ID {track_id}", (x1, max(22, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(annotated, f"Time: {timestamp:.2f}s", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(annotated)
                analyzed_frames += 1
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    unique_counts = {}
    unique_hazards = []
    for track in tracks:
        unique_counts[track["hazard"]] = unique_counts.get(track["hazard"], 0) + 1
        unique_hazards.append({
            "track_id": track["track_id"],
            "hazard": track["hazard"],
            "first_seen_seconds": round(track["first_seen"], 2),
            "last_seen_seconds": round(track["last_seen"], 2),
            "observations": track["observations"],
            "max_confidence": round(track["max_confidence"], 2),
        })

    summary = {
        "source_fps": source_fps,
        "source_frames": total_frames,
        "source_duration_seconds": round(duration, 2),
        "video_width": width,
        "video_height": height,
        "frame_step": frame_step,
        "sample_interval_seconds": sample_interval,
        "analyzed_frames": analyzed_frames,
        "unique_hazards": unique_hazards,
        "unique_hazard_counts": unique_counts,
        "unique_hazard_total": len(tracks),
        "frame_level_detection_total": len(detections),
        "tracking_method": "Same-class IoU and normalized center-distance association",
    }
    return output_path, detections, summary

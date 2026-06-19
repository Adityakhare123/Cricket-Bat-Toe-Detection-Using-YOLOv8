import cv2
from ultralytics import YOLO


# ============================================================
# DEFAULT COLORS
# ============================================================
# OpenCV uses BGR format
#
# Bat = Blue
# Toe = Green
# Unknown = Red
# ============================================================

DEFAULT_COLORS = {
    "Bat": (255, 0, 0),       # Blue
    "Toe": (0, 255, 0),       # Green
    "Unknown": (0, 0, 255)    # Red
}


# ============================================================
# FALLBACK CLASS MAPPING
# ============================================================
# Your deployed model is sometimes predicting the full bat as Toe.
# So mapping alone is not enough.
#
# We still keep fallback mapping here, but final correction happens
# in resolve_detection_labels().
# ============================================================

FALLBACK_CLASS_NAMES = {
    2: "Bat",
    1: "Toe",
    3: "Toe"
}


def get_class_name(cls_id, class_names):
    """
    Get class name using priority:

    1. Class mapping selected in Streamlit app.
    2. Fallback mapping.
    3. Unknown.
    """

    if class_names and cls_id in class_names:
        return class_names[cls_id]

    if cls_id in FALLBACK_CLASS_NAMES:
        return FALLBACK_CLASS_NAMES[cls_id]

    return f"Unknown {cls_id}"


def resolve_detection_labels(detections):
    """
    Fix wrong model behavior.

    Problem:
    Model sometimes predicts both:
    - full bat area as Toe
    - actual toe area as Toe

    Fix:
    If no Bat is detected but multiple Toe boxes are detected,
    promote the largest Toe box to Bat.
    """

    bat_detections = [
        d for d in detections
        if d["class_name"] == "Bat"
    ]

    toe_detections = [
        d for d in detections
        if d["class_name"] == "Toe"
    ]

    if len(bat_detections) == 0 and len(toe_detections) >= 2:
        largest_toe = max(
            toe_detections,
            key=lambda d: d["area"]
        )

        largest_toe["class_name"] = "Bat"

    return detections


def get_color(class_name):
    """
    Get bounding box color by class name.
    """

    if class_name == "Bat":
        return DEFAULT_COLORS["Bat"]

    if class_name == "Toe":
        return DEFAULT_COLORS["Toe"]

    return DEFAULT_COLORS["Unknown"]


def draw_label(frame, label, x1, y1, color):
    """
    Draw readable label with filled background.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, min(int(x1), frame_w - 1))
    y1 = max(0, min(int(y1), frame_h - 1))

    text_size, _ = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    text_w, text_h = text_size

    label_y = max(y1 - 10, text_h + 12)
    label_y = max(text_h + 12, min(label_y, frame_h - 1))

    bg_x1 = x1
    bg_y1 = max(0, label_y - text_h - 10)
    bg_x2 = min(frame_w - 1, x1 + text_w + 14)
    bg_y2 = min(frame_h - 1, label_y + 7)

    cv2.rectangle(
        frame,
        (bg_x1, bg_y1),
        (bg_x2, bg_y2),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 7, label_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_detection(frame, detection):
    """
    Draw one detection on a video frame.
    """

    class_name = detection["class_name"]
    confidence = detection["confidence"]

    x1 = detection["x1"]
    y1 = detection["y1"]
    x2 = detection["x2"]
    y2 = detection["y2"]

    color = get_color(class_name)
    label = f"{class_name} {confidence:.2f}"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        4
    )

    draw_label(
        frame=frame,
        label=label,
        x1=x1,
        y1=y1,
        color=color
    )


def process_video(
    model_path: str,
    input_video_path: str,
    output_video_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    class_names=None
):
    """
    Process input video using YOLOv8 and save annotated output video.
    """

    if class_names is None:
        class_names = {}

    model = YOLO(model_path)

    print("Loaded model names:", model.names)
    print("Using app class names:", class_names)
    print("Using fallback class names:", FALLBACK_CLASS_NAMES)

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None or fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {output_video_path}")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            verbose=False
        )

        result = results[0]
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                x1 = max(0, min(int(x1), frame_w - 1))
                y1 = max(0, min(int(y1), frame_h - 1))
                x2 = max(0, min(int(x2), frame_w - 1))
                y2 = max(0, min(int(y2), frame_h - 1))

                width_box = max(0, x2 - x1)
                height_box = max(0, y2 - y1)
                area = width_box * height_box

                class_name = get_class_name(
                    cls_id=cls_id,
                    class_names=class_names
                )

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "area": area
                    }
                )

        detections = resolve_detection_labels(detections)

        for detection in detections:
            draw_detection(frame, detection)

        out.write(frame)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()

    print(f"Video saved successfully at: {output_video_path}")
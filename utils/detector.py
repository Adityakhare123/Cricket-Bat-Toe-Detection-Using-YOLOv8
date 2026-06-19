from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO


class CricketBatToeDetector:
    """
    YOLOv8 detector for Cricket Bat and Toe detection.
    This class is used for image detection in Streamlit.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)

        print("Loaded model names:", self.model.names)

        # =====================================================
        # DEFAULT FALLBACK CLASS MAPPING
        # =====================================================
        # Current deployed model behavior:
        #
        # class 2 = Bat
        # class 1 = Toe
        # class 3 = Toe
        #
        # This fallback is used only if app.py does not pass
        # class_names or if the selected mapping misses a class.
        # =====================================================

        self.default_class_names = {
            2: "Bat",
            1: "Toe",
            3: "Toe"
        }

        # =====================================================
        # RGB COLORS
        # =====================================================
        # Streamlit displays images in RGB format.
        #
        # Bat = Blue
        # Toe = Green
        # Unknown = Red
        # =====================================================

        self.class_colors = {
            "Bat": (59, 130, 246),      # Blue
            "Toe": (34, 197, 94),       # Green
            "Unknown": (239, 68, 68)    # Red
        }

    def get_class_name(
        self,
        cls_id: int,
        class_names: Optional[Dict[int, str]] = None
    ) -> str:
        """
        Get class name using priority:

        1. Class mapping selected from Streamlit app.
        2. Default fallback mapping.
        3. Unknown.
        """

        if class_names and cls_id in class_names:
            return class_names[cls_id]

        if cls_id in self.default_class_names:
            return self.default_class_names[cls_id]

        return f"Unknown {cls_id}"

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get RGB color for class name.
        """

        if class_name == "Bat":
            return self.class_colors["Bat"]

        if class_name == "Toe":
            return self.class_colors["Toe"]

        return self.class_colors["Unknown"]

    def draw_label(
        self,
        image: np.ndarray,
        label: str,
        x1: int,
        y1: int,
        color: Tuple[int, int, int]
    ) -> None:
        """
        Draw label with filled background.
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2

        image_h, image_w = image.shape[:2]

        x1 = max(0, min(int(x1), image_w - 1))
        y1 = max(0, min(int(y1), image_h - 1))

        text_size, _ = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        text_w, text_h = text_size

        label_y = max(y1 - 10, text_h + 12)
        label_y = max(text_h + 12, min(label_y, image_h - 1))

        bg_x1 = x1
        bg_y1 = max(0, label_y - text_h - 10)
        bg_x2 = min(image_w - 1, x1 + text_w + 14)
        bg_y2 = min(image_h - 1, label_y + 7)

        cv2.rectangle(
            image,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1
        )

        cv2.putText(
            image,
            label,
            (x1 + 7, label_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    def predict_image(
        self,
        image: Image.Image,
        conf: float = 0.25,
        iou: float = 0.45,
        class_names: Optional[Dict[int, str]] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Run YOLOv8 prediction on uploaded image.

        Args:
            image: PIL image from Streamlit uploader.
            conf: Confidence threshold.
            iou: IOU threshold.
            class_names: Optional class mapping from Streamlit UI.

        Returns:
            annotated_image: RGB image with boxes and labels.
            detections: list of detection details.
        """

        image_np = np.array(image)

        results = self.model.predict(
            source=image_np,
            conf=conf,
            iou=iou,
            verbose=False
        )

        result = results[0]
        annotated_image = image_np.copy()
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return annotated_image, detections

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            image_h, image_w = annotated_image.shape[:2]

            x1 = max(0, min(int(x1), image_w - 1))
            y1 = max(0, min(int(y1), image_h - 1))
            x2 = max(0, min(int(x2), image_w - 1))
            y2 = max(0, min(int(y2), image_h - 1))

            class_name = self.get_class_name(
                cls_id=cls_id,
                class_names=class_names
            )

            color = self.get_color(class_name)
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                color,
                4
            )

            self.draw_label(
                image=annotated_image,
                label=label,
                x1=x1,
                y1=y1,
                color=color
            )

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            )

        return annotated_image, detections

    def detections_to_dataframe(self, detections: List[Dict]) -> pd.DataFrame:
        """
        Convert detection list to DataFrame.
        """

        if not detections:
            return pd.DataFrame(
                columns=[
                    "class_id",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2"
                ]
            )

        return pd.DataFrame(detections)
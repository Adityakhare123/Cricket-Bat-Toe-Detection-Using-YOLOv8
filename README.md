# Cricket Bat & Toe Detection Using YOLOv8

A custom real-time object detection system built with YOLOv8 to detect cricket bats and toes in cricket footage with high precision, optimized for sports video analysis, debugging, and model interpretability.

![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset%20Annotation-purple)

---

## Overview

This project focuses on detecting cricket bats and toes from real-world cricket images and video footage using a custom-trained YOLOv8 object detection model.

The goal was to build a reliable multi-class detection pipeline capable of distinguishing between visually similar objects, especially cricket bats and toes, which can often appear close together in gameplay frames.

The model was trained on a manually annotated dataset of approximately **9,000 images**, labeled using Roboflow. Through optimized training, IoU tuning, and Non-Maximum Suppression adjustments, the system achieved strong detection performance and reduced false positives.

---

## Key Results

- Achieved **92% mAP** on the custom cricket bat and toe detection dataset.
- Reduced false positives by **25%** using optimized IoU and NMS tuning.
- Improved class-wise detection precision by **35%** for visually similar classes.
- Built a manually labeled dataset of approximately **9,000 images**.
- Delivered real-time detection visualizations using OpenCV.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8 | Object detection model |
| Ultralytics | YOLOv8 training and inference framework |
| OpenCV | Real-time image and video visualization |
| Roboflow | Image annotation and dataset management |
| NumPy | Numerical operations |
| PyTorch | Deep learning backend used by YOLOv8 |
| Matplotlib | Training result visualization |
| Google Colab / Local GPU | Model training environment |

---

## Features

### Custom YOLOv8 Detection Pipeline

Designed and trained a YOLOv8-based object detection pipeline specifically for cricket footage.

The model detects:

- Cricket bat
- Toe

---

### Manually Labeled Dataset

Created a high-quality annotated dataset using Roboflow with approximately **9,000 labeled images**.

The dataset was prepared with precise bounding boxes to improve object localization and reduce confusion between visually similar objects.

---

### Multi-Class Object Detection

Fine-tuned a multi-class detection model to distinguish between cricket bats and toes, even when they appear close together or overlap in real-world cricket frames.

---

### False Positive Reduction

Optimized object detection confidence thresholds, IoU settings, and Non-Maximum Suppression logic to reduce incorrect detections.

This helped reduce false positives by approximately **25%**.

---

### Real-Time Visualization

Integrated OpenCV-based visualizations to display model predictions on images and video frames.

This made it easier to debug model behavior and explain results to non-technical stakeholders.

---

## Model Performance

| Metric | Result |
|---|---|
| mAP | 92% |
| False Positive Reduction | 25% |
| Class-Wise Precision Improvement | 35% |
| Dataset Size | ~9,000 images |
| Model Type | YOLOv8 Multi-Class Detection |

---

## Project Workflow

### 1. Data Collection

Collected cricket-related images and video frames containing bats, toes, and similar visual patterns.

The dataset included different angles, lighting conditions, player positions, and gameplay scenarios to improve model generalization.

---

### 2. Data Annotation

Used Roboflow to manually annotate approximately **9,000 images**.

Each image was labeled with bounding boxes for the required object classes:

```text
bat
toe
```

---

### 3. Dataset Preparation

Prepared the dataset in YOLO format with train, validation, and test splits.

Typical YOLO dataset structure:

```text
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

---

### 4. Model Training

Trained a YOLOv8 model using the custom cricket bat and toe dataset.

Example training command:

```bash
yolo detect train \
  data=data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16
```

---

### 5. Model Evaluation

Evaluated the model using standard object detection metrics such as:

- mAP
- Precision
- Recall
- Confusion matrix
- Class-wise confidence scores

---

### 6. Inference and Visualization

Used OpenCV to visualize predictions on cricket footage.

The model draws bounding boxes, class labels, and confidence scores on detected objects.

Example inference command:

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=input_video.mp4 \
  conf=0.25
```

---

## Installation

### Prerequisites

Make sure you have the following installed:

```text
Python 3.8+
pip
OpenCV
Ultralytics YOLOv8
PyTorch
Roboflow account
```

---

### Clone the Repository

```bash
git clone https://github.com/Adityakhare123/Cricket-Bat-Toe-Detection-Using-YOLOv8.git
cd Cricket-Bat-Toe-Detection-Using-YOLOv8
```

---

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install ultralytics opencv-python numpy matplotlib roboflow
```

Or, if using a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Dataset Configuration

Create a `data.yaml` file for YOLOv8 training.

```yaml
train: dataset/train/images
val: dataset/valid/images
test: dataset/test/images

nc: 2

names:
  - bat
  - toe
```

---

## Training the Model

Run the following command to train the YOLOv8 model:

```bash
yolo detect train \
  data=data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=cricket_bat_toe_detector
```

You can also use a larger YOLOv8 model for better accuracy:

```bash
yolo detect train \
  data=data.yaml \
  model=yolov8s.pt \
  epochs=100 \
  imgsz=640 \
  batch=16
```

---

## Running Inference

### Image Inference

```bash
yolo detect predict \
  model=runs/detect/cricket_bat_toe_detector/weights/best.pt \
  source=test_image.jpg \
  conf=0.25
```

---

### Video Inference

```bash
yolo detect predict \
  model=runs/detect/cricket_bat_toe_detector/weights/best.pt \
  source=input_video.mp4 \
  conf=0.25
```

---

### Webcam Inference

```bash
yolo detect predict \
  model=runs/detect/cricket_bat_toe_detector/weights/best.pt \
  source=0 \
  conf=0.25
```

---

## OpenCV Visualization Example

```python
from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/cricket_bat_toe_detector/weights/best.pt")

video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.25, iou=0.45)

    annotated_frame = results[0].plot()

    cv2.imshow("Cricket Bat & Toe Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Optimization Techniques

The model performance was improved using:

- IoU threshold tuning
- Non-Maximum Suppression optimization
- Confidence threshold adjustment
- Class-wise precision analysis
- Improved annotation quality
- Dataset balancing
- Real-world footage validation

---

## Challenges

### Bat and Toe Visual Similarity

Cricket bats and toes can appear close together in video frames, especially near the crease area. This made classification challenging.

The issue was addressed through better annotations, class-specific training, and model tuning.

---

### False Positives

Initial model versions produced false positives in complex frames.

This was improved by tuning confidence thresholds, IoU values, and NMS behavior.

---

### Real-World Footage Variability

Cricket footage can include motion blur, lighting variation, different camera angles, shadows, and overlapping objects.

The dataset was improved with diverse samples to make the model more robust.

---

## Folder Structure

```text
Cricket-Bat-Toe-Detection-Using-YOLOv8/
│
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   │
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   │
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   │
│   └── data.yaml
│
├── runs/
│   └── detect/
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Example `requirements.txt`

```text
ultralytics
opencv-python
numpy
matplotlib
roboflow
torch
torchvision
```

---

## Recommended `.gitignore`

```gitignore
# YOLO output folders
runs/
weights/

# Model files
*.pt
*.onnx
*.engine

# Large media files
*.avi
*.mp4
*.mov
*.mkv

# Dataset folders
dataset/
datasets/
data/
images/
labels/

# Python cache
__pycache__/
*.pyc

# Virtual environment
venv/
.env
```

---

## Future Improvements

- Add more object classes related to cricket gameplay.
- Improve detection under motion blur and low-light conditions.
- Deploy the model as a real-time web application.
- Add automated video highlight generation.
- Integrate model output with sports analytics dashboards.
- Experiment with larger YOLOv8 variants for improved accuracy.
- Optimize model for edge deployment.

---

## Use Cases

- Cricket sports analytics
- Automated cricket footage analysis
- Bat tracking
- Player movement analysis
- Training and coaching assistance
- Real-time match visualization
- Computer vision portfolio project

---

## Notes

Model weights, full datasets, and large output videos may not be included in this repository due to GitHub file size limitations.

Generated YOLO output folders such as `runs/` are usually excluded from version control.

---

## Author

**Aditya Khare**

---

## License

This project is intended for educational, research, and portfolio purposes.

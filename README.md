# 🏏 Cricket Bat Toe Detection Using YOLOv8

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:020617,50:22c55e,100:3b82f6&height=220&section=header&text=Cricket%20Bat%20Toe%20Detection&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=YOLOv8%20%7C%20OpenCV%20%7C%20Streamlit%20%7C%20Computer%20Vision&descAlignY=60&descAlign=50&descSize=18" />
</p>

<p align="center">
  <a href="https://cricket-bat-toe-detection-using-yolov8-oxpcgapflikgpdy7tt9zva.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20App-22c55e?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-111827?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

<p align="center">
  A computer vision web application that detects <b>cricket bats</b> and localizes the <b>bat toe region</b> from images and videos using YOLOv8.
</p>

---

## 🚀 Live Project

<p align="center">
  <a href="https://cricket-bat-toe-detection-using-yolov8-oxpcgapflikgpdy7tt9zva.streamlit.app/">
    <img src="https://img.shields.io/badge/Click%20Here%20to%20Try%20Live%20App-22c55e?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
</p>

🔗 **Live App:**
https://cricket-bat-toe-detection-using-yolov8-oxpcgapflikgpdy7tt9zva.streamlit.app/

---

## 📌 Project Overview

This project is designed to detect two important regions from cricket bat images and videos:

* 🏏 **Cricket Bat**
* 🎯 **Cricket Bat Toe**

The application allows users to upload an image or video, run YOLOv8 object detection, view the detected output, and download the final processed result.

It also includes a smart post-processing rule to fix cases where the full bat is mistakenly predicted as the toe region.

---

## ✨ Key Features

* 🖼️ Image detection support
* 🎥 Video detection support
* 🏏 Cricket bat localization
* 🎯 Cricket bat toe region localization
* 📦 YOLOv8 custom model integration
* ⚙️ Adjustable confidence threshold
* 🎚️ Adjustable IOU threshold
* 🧠 Smart post-processing correction
* 📥 Download processed image output
* 📥 Download processed video output
* 🌐 Streamlit Cloud deployment
* 🤗 Hugging Face model hosting
* 🎨 Premium dark UI design

---

## 🧰 Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,opencv,github,git,vscode" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-111827?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging%20Face-Model%20Hosting-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/NumPy-Data%20Processing-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-DataFrames-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</p>

---

## 📂 Updated Project Structure

```text
Cricket-Bat-Toe-Detection-Using-YOLOv8/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── utils/
│   ├── __init__.py
│   ├── detector.py
│   ├── video_utils.py
│   └── model_loader.py
│
├── models/
│   └── cricket_bat_toe_best.pt
│
├── outputs/
│   ├── image_detection_output.jpg
│   └── video_detection_output.mp4
│
├── notebooks/
│   └── Cricket_toe_v3.ipynb
│
├── assets/
│   ├── sample_input.png
│   ├── sample_output.png
│   └── demo_preview.png
│
└── runs/
    └── detect/
```

> `models/`, `outputs/`, and `runs/` are ignored in Git because they can contain large model files, generated videos, and prediction outputs.

---

## 🧠 Model Details

The custom detection model was trained using **YOLOv8**.

Model file:

```text
cricket_bat_toe_best.pt
```

The model is hosted on Hugging Face and downloaded automatically during deployment using the Streamlit secret:

```toml
MODEL_URL = "https://huggingface.co/AdityaKhare123/cricket-bat-toe-detection/resolve/main/cricket_bat_toe_best.pt"
```

---

## 🏷️ Class Mapping

Current recommended deployed class mapping:

```text
2 = Bat
1 = Toe
3 = Toe
```

The Streamlit sidebar allows changing class mapping if needed.

---

## 🎨 Detection Colors

| Object    | Box Color |
| --------- | --------- |
| 🏏 Bat    | Blue      |
| 🎯 Toe    | Green     |
| ❓ Unknown | Red       |

---

## ⚙️ How the App Works

```text
Upload Image/Video
        ↓
YOLOv8 Inference
        ↓
Class Mapping
        ↓
Smart Post-Processing
        ↓
OpenCV Bounding Boxes
        ↓
Display + Download Output
```

---

## 🧩 Smart Post-Processing Logic

Sometimes the trained model may detect both the full cricket bat and the actual toe region as `Toe`.

To improve the final output, this project applies a correction rule:

```text
If no Bat is detected
and multiple Toe boxes are detected,
then the largest Toe box is corrected to Bat.
```

This helps make the output cleaner and more useful for demo and portfolio purposes.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Adityakhare123/Cricket-Bat-Toe-Detection-Using-YOLOv8.git
cd Cricket-Bat-Toe-Detection-Using-YOLOv8
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 📋 Requirements

```txt
streamlit
ultralytics
opencv-python-headless
numpy
pandas
matplotlib
pillow
torch
torchvision
requests
```

---

## 🚀 Deployment

This project is deployed using **Streamlit Cloud**.

Deployment configuration:

```text
Repository: Adityakhare123/Cricket-Bat-Toe-Detection-Using-YOLOv8
Branch: main
Main file path: app.py
Python version: 3.11
```

Streamlit secret:

```toml
MODEL_URL = "https://huggingface.co/AdityaKhare123/cricket-bat-toe-detection/resolve/main/cricket_bat_toe_best.pt"
```

Live app:

```text
https://cricket-bat-toe-detection-using-yolov8-oxpcgapflikgpdy7tt9zva.streamlit.app/
```

---

## 🖼️ Image Detection Workflow

1. Open the live app.
2. Go to the **Image Detection** tab.
3. Upload a `.jpg`, `.jpeg`, or `.png` image.
4. Click **Run Image Detection**.
5. View and download the output image.

---

## 🎥 Video Detection Workflow

1. Open the live app.
2. Go to the **Video Detection** tab.
3. Upload a `.mp4`, `.avi`, `.mov`, or `.mkv` video.
4. Click **Run Video Detection**.
5. Wait for frame-by-frame processing.
6. View and download the output video.

---

## 📌 Current Capability

The current app can:

* Detect cricket bats
* Detect cricket bat toe region
* Process images
* Process videos
* Apply custom class mapping
* Correct common bat/toe misclassification
* Generate downloadable outputs

---

## 🔮 Future Improvements

* Improve dataset quality
* Add more cricket bat images under different lighting
* Retrain model with cleaner toe annotations
* Add toe guard detection
* Add crack and damage detection
* Add real-time webcam detection
* Add class-wise confidence filters
* Improve bat boundary accuracy
* Add model evaluation dashboard
* Add sample demo images inside README

---

## 👨‍💻 Author

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=soft&color=0:020617,100:22c55e&height=120&section=footer&text=Aditya%20Khare&fontSize=36&fontColor=ffffff&animation=fadeIn" />
</p>

**Aditya Khare**

<p>
  <a href="https://github.com/Adityakhare123">
    <img src="https://img.shields.io/badge/GitHub-Adityakhare123-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
  <a href="https://cricket-bat-toe-detection-using-yolov8-oxpcgapflikgpdy7tt9zva.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Project-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
</p>

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub.

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,50:22c55e,100:020617&height=120&section=footer" />
</p>

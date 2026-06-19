# 🏏 Cricket Bat Toe Detection Using YOLOv8

A computer vision project built using **YOLOv8**, **OpenCV**, and **Streamlit** to detect cricket bats and localize the **toe region** of the bat from images and videos.

## 🚀 Live Demo

🔗 **Try the deployed app here:**
https://cricket-bat-toe-detection-using-yolov8-oxpcgapflikgpdy7tt9zva.streamlit.app/

---

## 📌 Project Overview

This project focuses on detecting two key regions from cricket bat media:

* **Cricket Bat**
* **Cricket Bat Toe**

The application supports both **image detection** and **video detection**. Users can upload an image or video, run YOLOv8 inference, and download the processed output with bounding boxes.

The project also includes a smart post-processing fix where, if the model predicts the full bat as a toe region, the largest detected toe box is corrected and treated as the bat.

---

## ✨ Features

* Detect cricket bat from images
* Detect cricket bat toe region
* Upload and process videos
* Download processed image output
* Download processed video output
* Streamlit-based premium UI
* YOLOv8 object detection backend
* OpenCV video frame processing
* Custom class mapping support
* Smart post-processing correction
* Deployed on Streamlit Cloud
* Model hosted externally using Hugging Face

---

## 🛠️ Tech Stack

* **Python**
* **YOLOv8**
* **Ultralytics**
* **OpenCV**
* **Streamlit**
* **NumPy**
* **Pandas**
* **Pillow**
* **Hugging Face**
* **Streamlit Cloud**

---

## 📂 Project Structure

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

> Note: The `models/`, `outputs/`, and `runs/` folders are ignored in Git because they can contain large files. The trained model is loaded using a Hugging Face download link during deployment.

---

## 🧠 Model Information

The object detection model was trained using **YOLOv8**.

Current model file:

```text
cricket_bat_toe_best.pt
```

The model is hosted on Hugging Face and downloaded automatically by the Streamlit app using the `MODEL_URL` environment variable.

Recommended class mapping:

```text
2 = Bat
1 = Toe
3 = Toe
```

---

## 🎨 Detection Colors

| Object  | Color |
| ------- | ----- |
| Bat     | Blue  |
| Toe     | Green |
| Unknown | Red   |

---

## ⚙️ How It Works

1. User uploads an image or video.
2. YOLOv8 model runs object detection.
3. Detected class IDs are mapped to labels.
4. OpenCV draws bounding boxes and labels.
5. For videos, each frame is processed one by one.
6. Smart post-processing corrects cases where the full bat is wrongly detected as toe.
7. Final output is displayed and can be downloaded.

---

## 🧩 Smart Post-Processing Logic

Sometimes the trained model may detect both the full bat and the toe as `Toe`.

To handle this, the project includes a correction rule:

```text
If no Bat is detected,
and multiple Toe boxes are detected,
then the largest Toe box is corrected to Bat.
```

This improves the final output visually and makes the detection more practical for demo usage.

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

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

Usually:

```text
http://localhost:8501
```

---

## 🔐 Environment Variable

For deployment, the model is loaded using a Hugging Face direct download link.

Create a Streamlit secret:

```toml
MODEL_URL = "https://huggingface.co/AdityaKhare123/cricket-bat-toe-detection/resolve/main/cricket_bat_toe_best.pt"
```

---

## 📋 requirements.txt

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

Deployment settings:

```text
Repository: Adityakhare123/Cricket-Bat-Toe-Detection-Using-YOLOv8
Branch: main
Main file path: app.py
Python version: 3.11
```

Streamlit secrets:

```toml
MODEL_URL = "https://huggingface.co/AdityaKhare123/cricket-bat-toe-detection/resolve/main/cricket_bat_toe_best.pt"
```

Live app:

```text
https://cricket-bat-toe-detection-using-yolov8-oxpcgapflikgpdy7tt9zva.streamlit.app/
```

---

## 🧪 Usage

### Image Detection

1. Open the app.
2. Go to **Image Detection** tab.
3. Upload a `.jpg`, `.jpeg`, or `.png` image.
4. Click **Run Image Detection**.
5. View and download the output image.

### Video Detection

1. Go to **Video Detection** tab.
2. Upload a `.mp4`, `.avi`, `.mov`, or `.mkv` video.
3. Click **Run Video Detection**.
4. Wait for processing.
5. View and download the output video.

---

## 📌 Current Capability

The current version can:

* Detect cricket bat region
* Detect cricket bat toe region
* Process images
* Process videos
* Apply custom class mapping
* Correct common bat/toe misclassification using post-processing

---

## 🔮 Future Improvements

* Improve dataset quality
* Retrain model with more toe examples
* Add toe guard detection
* Add crack and damage detection
* Add real-time webcam detection
* Add confidence-based filtering per class
* Improve bat boundary accuracy
* Add evaluation metrics to the app
* Add sample demo images to README

---

## 👨‍💻 Author

**Aditya Khare**

GitHub:
https://github.com/Adityakhare123

Live Project:
https://cricket-bat-toe-detection-using-yolov8-oxpcgapflikgpdy7tt9zva.streamlit.app/

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub.

---

## 📄 License

This project is created for learning, demonstration, and portfolio purposes.

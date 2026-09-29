# 🥭 Mango Multiclass Disease Detection

An AI-based mango leaf disease detection system that uses **deep learning and computer vision** to identify diseases present on mango leaves and localize affected regions using bounding boxes.

The system is designed to handle both **single-disease and multi-disease mango leaves**, providing disease predictions along with the location of detected disease regions.

---

## 📌 Overview

Mango crops can be affected by multiple diseases that reduce crop quality and yield. Manual identification of diseases from leaf symptoms can be difficult, especially when multiple diseases occur on the same leaf.

This project uses computer vision and deep learning to automatically analyze mango leaf images and:

* Detect disease-affected regions
* Identify the corresponding disease classes
* Localize detected diseases using bounding boxes
* Support multiple disease detections on a single leaf
* Provide confidence scores for predictions

The current detection pipeline is based on a **YOLO object-detection model trained using annotated mango leaf images**.

---

## 🚀 Features

* 🌿 Mango leaf disease detection
* 🔍 Disease localization using bounding boxes
* 🦠 Multi-class disease detection
* 🦠 Multiple disease detection on a single leaf
* 📊 Confidence scores for predictions
* 🖼️ Image-based inference
* ⚡ REST API for model inference
* 🧠 Deep-learning based computer vision
* 📱 Web-based interface
* ☁️ Deployable frontend and backend architecture

---

## 🧠 Technology Stack

### Machine Learning / Computer Vision

* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* NumPy
* Pandas
* Matplotlib

### Backend

* Python
* Flask
* REST API
* Gunicorn

### Frontend

* React.js
* JavaScript
* HTML
* CSS

### Deployment

* Netlify — Frontend
* Render — Backend

### Development

* Git
* GitHub
* Google Colab
* VS Code

---

## 🏗️ System Architecture

```text
                 ┌──────────────────────┐
                 │     User Uploads     │
                 │    Mango Leaf Image  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    React Frontend    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     REST API         │
                 │   Python Backend     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     YOLOv8 Model     │
                 │ Disease Detection &  │
                 │    Localization      │
                 └──────────┬───────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ Disease + Confidence +     │
              │ Bounding Box Coordinates   │
              └─────────────┬──────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Frontend Result   │
                 │   Visualization      │
                 └──────────────────────┘
```

---

## 📂 Project Structure

```text
Mango_Multiclass_Disease_Detection/
│
├── backend/
│   ├── models/
│   │   └── trained_model.pt
│   │
│   ├── routes/
│   ├── services/
│   ├── utils/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── training/
│   ├── train_yolov8.py
│   ├── dataset.yaml
│   └── ...
│
├── datasets/
│   └── README.md
│
├── README.md
└── .gitignore
```

> The exact folder structure may differ depending on the current project implementation.

---

# 🧪 Dataset

The project uses an **annotated mango leaf dataset prepared for YOLO object detection**.

Each training image is associated with YOLO-format annotations containing:

```text
class_id center_x center_y width height
```

All bounding-box coordinates are normalized between `0` and `1`.

### Dataset preparation includes:

* Image validation
* Annotation validation
* Class ID verification
* Bounding-box validation
* Duplicate checking
* Corrupted-image checking
* Train/validation/test splitting
* Class distribution analysis

### Important

The project may contain additional datasets such as Kaggle data or previous experimental datasets.

These datasets are **not automatically considered for the current training pipeline**.

The active training pipeline should use only the **verified YOLO-annotated dataset**.

Other datasets are retained for reference/backup and are excluded from the current training process.

---

# 🏷️ YOLO Annotation Format

Each image has a corresponding `.txt` annotation file.

Example:

```text
image_001.jpg
image_001.txt
```

Example annotation:

```text
0 0.512 0.438 0.274 0.316
1 0.721 0.604 0.182 0.221
```

Each row represents:

```text
class_id
center_x
center_y
width
height
```

For a multi-disease leaf, multiple annotation rows can represent different disease regions.

---

# 🧠 Model

## YOLOv8

The primary detection model is **YOLOv8** from Ultralytics.

YOLOv8 performs:

* Object detection
* Disease classification
* Disease localization
* Multi-object detection in a single image

For example, if a leaf contains two genuinely annotated disease regions, the model can return:

```text
Disease A
Confidence: 91%
Bounding Box: (...)

Disease B
Confidence: 87%
Bounding Box: (...)
```

The model should not force a disease prediction when there is no valid detection.

---

# 🔬 Training

The model can be trained using Google Colab or another GPU-enabled environment.

Typical training workflow:

```text
Dataset
   ↓
Annotation Validation
   ↓
Train / Validation / Test Split
   ↓
YOLOv8 Training
   ↓
Validation
   ↓
Model Evaluation
   ↓
Best Model (.pt)
   ↓
Backend Inference
```

Example training command:

```bash
yolo detect train \
    data=dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16
```

The exact training parameters should be adjusted according to dataset size and available GPU resources.

---

# 📊 Model Evaluation

The model should be evaluated using:

* Precision
* Recall
* mAP@50
* mAP@50-95
* Confusion matrix
* Per-class performance
* Validation loss
* Test-set predictions

Example evaluation:

```bash
yolo detect val \
    model=best.pt \
    data=dataset.yaml
```

A confusion matrix should be reviewed to identify whether one disease is being incorrectly predicted more frequently than other classes.

---

# ⚠️ Preventing Model Bias

A major focus of the training pipeline is preventing the model from becoming biased toward a single disease class.

Before training, the dataset should be checked for:

* Severe class imbalance
* Incorrect class IDs
* Incorrect annotations
* Duplicate images
* Data leakage
* Corrupted images
* Empty annotation files
* Incorrect bounding boxes
* Inconsistent class names

For example, if the model predicts **Anthracnose for almost every image**, the dataset and training pipeline must be investigated before deploying the model.

The solution should not simply force another prediction or hardcode a class.

---

# 🖥️ Web Application

The web application allows users to upload a mango leaf image and receive detection results.

### Workflow

```text
Upload Image
     ↓
Backend API
     ↓
YOLOv8 Inference
     ↓
Disease Detection
     ↓
Bounding Box Localization
     ↓
Confidence Calculation
     ↓
Result Display
```

The frontend visualizes the detected disease regions directly on the uploaded image.

---

# 🔌 API

The backend exposes an API for image prediction.

Example request:

```http
POST /predict
Content-Type: multipart/form-data
```

Example response:

```json
{
  "predictions": [
    {
      "disease": "Anthracnose",
      "confidence": 0.91,
      "bbox": [120, 85, 310, 270]
    }
  ]
}
```

The exact response structure depends on the current backend implementation.

---

# 🛠️ Local Setup

## 1. Clone Repository

```bash
git clone https://github.com/adityayadvv45/Mango_Multiclass_Disease_Detection.git
cd Mango_Multiclass_Disease_Detection
```

---

## 2. Backend Setup

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
python app.py
```

---

## 3. Frontend Setup

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend will then be available at the local development URL shown by Vite.

---

# ☁️ Deployment

The project uses a separate frontend and backend architecture.

### Frontend

Deployed using:

**Netlify**

Project:

https://mango-multiclass-detection-system.netlify.app/

### Backend

Deployed using:

**Render**

Backend:

https://mango-multiclass-disease-detection-4.onrender.com/

---

# 🔮 Future Improvements

Possible future improvements include:

* Larger and more diverse real-world datasets
* Improved multi-disease detection
* Better class balancing
* More robust validation datasets
* Disease severity estimation
* Leaf segmentation
* Mobile application
* Offline inference
* Explainable AI using Grad-CAM or similar techniques
* Continuous model evaluation with new field images

---

# 🎯 Applications

This system can potentially assist with:

* Agricultural disease screening
* Mango crop monitoring
* Early disease identification
* Research in agricultural computer vision
* Smart agriculture applications
* Automated plant disease analysis

The system is intended as an **AI-assisted detection tool**, not a replacement for expert agricultural diagnosis.

---

# 👨‍💻 Author

**Aditya Yadav**

B.Tech Computer Science & Engineering

GitHub:
https://github.com/adityayadvv45

Portfolio:
https://aditya-portfolio-8zfw.vercel.app/

LinkedIn:
https://www.linkedin.com/in/aditya-yadav-289b132b3/

---

# 📄 License

This project is intended for educational, research, and demonstration purposes.


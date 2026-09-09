# 🥭 Mango Leaf Multiple Disease Detection

An AI-powered web application for detecting and analyzing diseases present on mango leaves using **Deep Learning, Computer Vision, CNN, and YOLOv8**.

The system provides an interactive React-based interface where users can upload mango leaf images and receive AI-based disease analysis with confidence information and visual localization where supported by the detection model.

---

## 🚀 Features

* 🌿 Mango leaf disease detection using Deep Learning
* 🧠 **CNN-based image classification**
* 🎯 **YOLOv8-based object detection/localization**
* 🔍 Computer vision processing using **OpenCV**
* 📊 Confidence-based prediction results
* 🖼️ Drag-and-drop image upload
* ⚡ Real-time communication between frontend and backend
* 📍 Disease localization using bounding boxes when supported by the YOLOv8 model
* 📚 Mango disease information and symptoms
* 💡 Disease-specific recommendations
* 📱 Responsive React interface
* 🔎 Disease search and filtering
* 🎨 Modern UI built using Tailwind CSS
* 🔌 REST API-based ML integration
* 🛡️ Image validation and error handling

---

# 🧠 AI / ML Architecture

The project combines **Deep Learning classification** and **Computer Vision object detection**.

```text
                    Mango Leaf Image
                           │
                           ▼
                 ┌──────────────────┐
                 │   React Frontend │
                 └────────┬─────────┘
                          │
                          │ HTTP Request
                          ▼
                 ┌──────────────────┐
                 │   Python API     │
                 │  FastAPI/REST    │
                 └────────┬─────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Image Preprocessing │
                │      OpenCV         │
                └─────────┬───────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      ┌──────────────┐          ┌──────────────┐
      │     CNN      │          │    YOLOv8    │
      │ Classification│          │  Detection   │
      └───────┬──────┘          └───────┬──────┘
              │                         │
              ▼                         ▼
       Disease Class             Disease Regions
       + Confidence              + Bounding Boxes
              │                         │
              └────────────┬────────────┘
                           ▼
                  ┌─────────────────┐
                  │ Prediction JSON │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ React Dashboard │
                  └─────────────────┘
```

---

# 🤖 Machine Learning Models

## CNN — Convolutional Neural Network

The CNN component is used for image-based disease classification.

CNNs are particularly suitable for image analysis because they can automatically learn visual patterns such as:

* Leaf texture
* Spots
* Discoloration
* Lesions
* Fungal patterns
* Surface abnormalities
* Disease-specific visual features

The CNN produces disease predictions along with confidence information based on the trained model.

---

## YOLOv8 — Object Detection

**YOLOv8** is used for visual disease detection and localization.

Unlike a simple classification model that produces a class prediction for an entire image, an object-detection model can identify **where a detected disease appears in the image**.

YOLOv8 can provide:

* Disease class
* Detection confidence
* Bounding-box coordinates
* Multiple detections in one image

Example:

```text
Mango Leaf
│
├── Anthracnose       92%
│      └── Bounding Box
│
├── Powdery Mildew    87%
│      └── Bounding Box
│
└── Bacterial Canker 81%
       └── Bounding Box
```

The actual number of simultaneously detectable diseases depends on the classes and annotations used to train the detection model.

---

# 👁️ Computer Vision

## OpenCV

**OpenCV (Open Source Computer Vision Library)** is used for image-processing operations in the ML pipeline.

It can be used for:

* Image reading
* Image resizing
* Image preprocessing
* Color-space conversion
* Image manipulation
* Bounding-box visualization
* Computer-vision preprocessing

---

# 🔧 Technology Stack

## Frontend

| Technology            | Purpose                   |
| --------------------- | ------------------------- |
| **React.js**          | Frontend UI               |
| **Vite**              | Frontend build tool       |
| **JavaScript (ES6+)** | Application logic         |
| **Tailwind CSS**      | Styling and responsive UI |
| **React Components**  | Modular UI architecture   |
| **Axios / Fetch API** | Backend communication     |

## Backend

| Technology             | Purpose                        |
| ---------------------- | ------------------------------ |
| **Python**             | Backend & ML development       |
| **FastAPI / REST API** | ML prediction API              |
| **Uvicorn**            | ASGI server                    |
| **Pydantic**           | API data validation            |
| **CORS**               | Frontend-backend communication |

## Artificial Intelligence

| Technology      | Purpose                              |
| --------------- | ------------------------------------ |
| **CNN**         | Disease classification               |
| **YOLOv8**      | Disease detection & localization     |
| **Ultralytics** | YOLOv8 implementation                |
| **PyTorch**     | Deep Learning framework used by YOLO |
| **OpenCV**      | Image processing                     |
| **NumPy**       | Numerical/image-array operations     |
| **Pandas**      | Dataset/data processing              |

---

# 🖥️ Frontend Architecture

The frontend is built using React + Vite and follows a component-based architecture.

```text
MangoLeaf/
│
├── src/
│   │
│   ├── components/
│   │   │
│   │   ├── common/
│   │   │   ├── Badge.jsx
│   │   │   ├── Button.jsx
│   │   │   └── Toast.jsx
│   │   │
│   │   ├── layout/
│   │   │   ├── Navbar.jsx
│   │   │   └── Footer.jsx
│   │   │
│   │   ├── home/
│   │   │   ├── Hero.jsx
│   │   │   ├── HowItWorks.jsx
│   │   │   ├── SupportedDiseases.jsx
│   │   │   └── QuickStats.jsx
│   │   │
│   │   ├── detection/
│   │   │   ├── UploadBox.jsx
│   │   │   ├── ImagePreview.jsx
│   │   │   ├── AnalysisLoader.jsx
│   │   │   ├── ResultDashboard.jsx
│   │   │   ├── PredictionBreakdown.jsx
│   │   │   ├── DiseaseInfo.jsx
│   │   │   └── RecommendationCard.jsx
│   │   │
│   │   └── diseases/
│   │       ├── DiseaseCard.jsx
│   │       └── DiseaseFilter.jsx
│   │
│   ├── data/
│   │   ├── diseases.js
│   │   ├── mockResults.js
│   │   └── sampleImages.js
│   │
│   ├── services/
│   │   └── predictionService.js
│   │
│   ├── pages/
│   │   ├── Home.jsx
│   │   ├── Detection.jsx
│   │   ├── Diseases.jsx
│   │   └── About.jsx
│   │
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
│
├── tailwind.config.js
├── vite.config.js
└── package.json
```

---

# 📁 Frontend Components

### `components/common/`

Reusable UI components.

* `Badge.jsx` — Category and risk-level badges
* `Button.jsx` — Primary, secondary and outline button variants
* `Toast.jsx` — Non-intrusive validation and notification messages

### `components/layout/`

Application-wide layout.

* `Navbar.jsx` — Sticky navigation and mobile menu
* `Footer.jsx` — Application footer and project information

### `components/home/`

Landing-page sections.

* `Hero.jsx` — Project introduction and interactive CNN preview
* `HowItWorks.jsx` — Three-step detection pipeline
* `SupportedDiseases.jsx` — Supported disease preview
* `QuickStats.jsx` — Technical/project statistics

### `components/detection/`

Core disease-detection interface.

* `UploadBox.jsx` — Drag-and-drop image upload
* `ImagePreview.jsx` — Uploaded image preview
* `AnalysisLoader.jsx` — AI analysis animation
* `ResultDashboard.jsx` — Detection results dashboard
* `PredictionBreakdown.jsx` — Confidence visualization
* `DiseaseInfo.jsx` — Disease symptoms and visual information
* `RecommendationCard.jsx` — Disease-management recommendations

### `components/diseases/`

Disease encyclopedia components.

* `DiseaseCard.jsx` — Disease information card
* `DiseaseFilter.jsx` — Search and category filtering

---

# 🔌 Prediction Service

```text
src/services/predictionService.js
```

This service provides a decoupled interface between the React frontend and the machine-learning backend.

```text
React Detection Page
        │
        ▼
predictionService.js
        │
        ▼
REST API
        │
        ▼
Python ML Backend
        │
        ▼
CNN / YOLOv8
```

This architecture makes it easier to replace or update the ML model without rewriting the complete frontend.

---

# 🐍 Backend Architecture

The backend contains the Python API and machine-learning inference pipeline.

Recommended logical structure:

```text
backend/
│
├── main.py / app.py
│
├── models/
│   ├── CNN model
│   └── YOLOv8 model
│
├── services/
│   ├── prediction service
│   ├── preprocessing
│   └── detection service
│
├── utils/
│   ├── image processing
│   └── response utilities
│
├── requirements.txt
└── ...
```

> The exact backend filenames may differ depending on the implementation in the project.

---

# 🔄 Prediction Workflow

```text
1. User uploads mango leaf image
              ↓
2. React validates the image
              ↓
3. Image sent to Python REST API
              ↓
4. Backend receives multipart image
              ↓
5. OpenCV performs preprocessing
              ↓
6. Trained ML model performs inference
              ↓
7. CNN / YOLOv8 generates prediction
              ↓
8. Confidence and detection information
   are extracted
              ↓
9. YOLOv8 bounding boxes are generated
   when available
              ↓
10. Backend returns JSON response
              ↓
11. React displays result dashboard
```

---

# 🌿 Disease Detection

The application is designed around multiple mango-leaf disease classes.

The exact disease classes are determined by the trained dataset/model configuration rather than being hardcoded by the frontend.

A detection response can contain information such as:

```json
{
  "predictions": [
    {
      "disease": "Disease Name",
      "confidence": 0.92
    }
  ],
  "detections": [
    {
      "disease": "Disease Name",
      "confidence": 0.92,
      "bbox": [120, 80, 350, 290]
    }
  ]
}
```

The frontend then converts this response into the visual result dashboard.

---

# 🎯 Multiple Disease Detection

A major objective of the project is to move beyond simple single-class classification.

If the trained detection model identifies multiple disease regions in the same leaf, the application can represent them separately:

```text
                 Mango Leaf
        ┌──────────────────────────┐
        │                          │
        │   ┌────────┐             │
        │   │Disease │             │
        │   │   A    │             │
        │   └────────┘             │
        │                          │
        │              ┌─────────┐ │
        │              │Disease  │ │
        │              │    B    │ │
        │              └─────────┘ │
        │                          │
        └──────────────────────────┘
```

This allows the system to provide both:

* **What disease was detected?**
* **Where was it detected?**

> Multiple simultaneous disease detection and localization require a model trained for the corresponding detection/multi-label task. The application does not artificially generate multiple diseases from a single-class softmax prediction.

---

# 📊 Result Dashboard

After inference, the application presents:

* Detected disease
* Confidence score
* Prediction breakdown
* Disease symptoms
* Visual indicators
* Recommended actions
* Detection locations/bounding boxes when available

Example:

```text
┌─────────────────────────────────────────┐
│           Detection Result              │
├─────────────────────────────────────────┤
│ Disease: Anthracnose                    │
│ Confidence: 92%                         │
│                                         │
│ ━━━━━━━━━━━━━━━━━━━━━ 92%               │
│                                         │
│ Symptoms                                │
│ • Dark lesions                          │
│ • Leaf discoloration                    │
│                                         │
│ Recommendation                          │
│ Follow appropriate cultural/agronomic   │
│ disease-management practices.           │
└─────────────────────────────────────────┘
```

---

# 🛠️ Installation

## Prerequisites

Make sure the following are installed:

* Node.js
* npm
* Python 3.x
* Git
* pip

---

## Clone Repository

```bash
git clone https://github.com/adityayadvv45/Mango_Multiclass_Disease_Detection.git
cd Mango_Multiclass_Disease_Detection
```

---

# 🎨 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The Vite development server will start locally.

---

# 🐍 Backend Setup

Open another terminal:

```bash
cd backend
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API server:

```bash
uvicorn main:app --reload
```

> Replace `main:app` with the actual module and FastAPI application name if your backend uses a different entry point.

---

# 🔗 API

The frontend communicates with the Python backend through a REST API.

Typical prediction request:

```http
POST /predict
Content-Type: multipart/form-data
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@mango_leaf.jpg"
```

The backend processes the image and returns prediction information to the frontend.

---

# 🧪 Testing

Test the system using mango leaf images containing different visual conditions.

Recommended test cases:

1. Healthy leaf
2. Single-disease leaf
3. Different disease classes
4. Images with multiple visible disease regions
5. Low-quality images
6. Invalid file types
7. Large images
8. Different lighting conditions

The goal is to verify that the model returns results based on the actual image rather than static/mock responses.

---

# ⚙️ Environment Variables

For production deployment, environment-specific values should be stored using environment variables rather than hardcoded values.

Example:

```env
VITE_API_URL=http://localhost:8000
```

For deployment:

```env
VITE_API_URL=https://your-backend-url
```

Never commit secrets or private API keys to GitHub.

---

# ☁️ Deployment

The project can be deployed using separate frontend and backend services.

### Frontend

Recommended:

* Vercel
* Netlify

### Backend / ML API

Recommended:

* Railway
* Render
* Docker-compatible cloud platforms

### Railway Architecture

```text
Railway Project
│
├── Frontend Service
│      └── React + Vite
│
└── Backend Service
       └── Python + FastAPI
              │
              ├── CNN
              ├── YOLOv8
              └── OpenCV
```

For production deployment, the backend must listen on the port supplied by the hosting platform.

---

# 📦 Dependencies

Important ML/backend dependencies include:

```text
Python
FastAPI
Uvicorn
Ultralytics
PyTorch
OpenCV
NumPy
Pandas
Pillow
```

Frontend dependencies include:

```text
React
Vite
Tailwind CSS
JavaScript
Axios / Fetch API
```

The exact dependency versions are maintained in the project's dependency files.

---

# 🔐 Git & Repository Hygiene

The Python virtual environment should **never** be committed.

The repository uses `.gitignore` to exclude files such as:

```text
.venv/
__pycache__/
*.pyc
node_modules/
dist/
.env
```

This keeps the repository lightweight and reproducible.

---

# 🎯 Project Objectives

The main objectives of the project are:

* Automate mango leaf disease detection
* Apply deep learning to agricultural image analysis
* Detect multiple disease classes
* Localize disease regions when supported by the detection model
* Provide confidence-based predictions
* Present complex ML results through a simple interface
* Build a practical full-stack AI application
* Create a foundation for future agricultural AI systems

---

# 🔮 Future Enhancements

* 📷 Real-time camera-based detection
* 📱 Mobile application
* 🌐 Multilingual disease information
* 🌱 More mango disease classes
* 📈 Model performance analytics
* 🗺️ Regional disease monitoring
* ☁️ Scalable cloud inference
* 🧠 Model optimization for faster CPU inference
* 📊 Dataset and model evaluation dashboard
* 🔔 Disease alerts and notifications
* 🌾 Advanced treatment and agricultural recommendations

---

# ⚠️ Disclaimer

This application is an AI-based educational/research project.

Predictions should not be considered a substitute for professional agricultural diagnosis. Model performance depends on the quality, diversity, and labeling of the training dataset.

---

# 👨‍💻 Author

**Aditya Yadav And  Abhishek Mishra **

B.Tech Computer Science Engineering

GitHub:
https://github.com/adityayadvv45/Mango_Multiclass_Disease_Detection

LinkedIn:
https://www.linkedin.com/in/aditya-yadav-289b132b3

---

# ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Mango Leaf Multiple Disease Detection — AI + Computer Vision + Full-Stack Web Application**


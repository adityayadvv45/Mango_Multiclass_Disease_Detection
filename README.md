# 🌿 Mango Leaf Multiple Disease Detection

An AI-powered web application for **detecting diseases in mango leaves using computer vision and deep learning**.

The system analyzes an uploaded mango leaf image, identifies the detected disease(s), provides confidence scores, and—when supported by the trained detection model—**highlights the affected regions directly on the leaf image**.

---

## 🚀 Features

* 🌿 Mango leaf disease detection
* 🔍 Multiple disease class support
* 🤖 Deep Learning based prediction
* 🎯 Disease localization using bounding boxes when supported
* 📊 Confidence score for predictions
* 📈 Disease probability/prediction details
* 🖼️ Image upload and preview
* ⚡ Fast ML inference through Python backend
* 🔗 React frontend connected with ML API
* 🛡️ Input validation and error handling
* 📱 Responsive and modern UI
* 🔄 Real-time prediction workflow

---

## 🧠 How It Works

The application follows an end-to-end AI pipeline:

```text
User
  │
  ▼
Upload Mango Leaf Image
  │
  ▼
React Frontend
  │
  │  HTTP Request
  ▼
Python ML Backend
  │
  ▼
Image Preprocessing
  │
  ▼
Deep Learning Model
  │
  ├── Disease Classification
  │
  └── Disease Detection / Localization
  │
  ▼
Prediction + Confidence + Location
  │
  ▼
JSON Response
  │
  ▼
React Result Interface
```

---

## 🔬 Machine Learning

The project uses deep learning and computer vision techniques for mango leaf analysis.

### ML Pipeline

```text
Input Image
     ↓
Image Validation
     ↓
Preprocessing
     ↓
ML Model
     ↓
Disease Prediction
     ↓
Confidence Calculation
     ↓
Disease Localization
     ↓
Final Result
```

The inference pipeline uses the **trained model and the same preprocessing/class mapping used during model development** to ensure that predictions correspond correctly to the trained classes.

---

## 🌱 Disease Detection

The application is designed to support **all disease classes available in the trained model**, rather than limiting the application to a fixed number of diseases.

The supported classes are loaded from the project's actual model/dataset configuration.

Typical mango leaf disease categories may include:

* Healthy
* Anthracnose
* Bacterial Canker
* Powdery Mildew
* Sooty Mold
* Die Back
* Gall Midge
* Cutting Weevil
* Mango Malformation

> **Note:** The exact classes supported by the application depend on the trained model used in this project.

---

## 🎯 Multiple Disease Detection

The system is designed to handle cases where multiple affected regions are present in a single leaf image, **provided the trained detection/model architecture supports region-level or multi-label detection**.

For example:

```text
Mango Leaf
     │
     ├── Region 1 → Anthracnose
     │
     ├── Region 2 → Powdery Mildew
     │
     └── Region 3 → Bacterial Canker
```

The result can contain:

```text
Anthracnose       92%
Powdery Mildew    86%
Bacterial Canker  81%
```

The application does not artificially generate multiple predictions. Results come from the actual trained ML pipeline.

---

## 📍 Disease Localization

When the detection model provides spatial coordinates, the application displays the affected regions directly on the uploaded image.

Example:

```text
┌───────────────────────────────┐
│                               │
│      ┌──────────────┐         │
│      │ Anthracnose  │         │
│      │     92%      │         │
│      └──────────────┘         │
│                               │
│                  ┌─────────┐  │
│                  │ Powdery │  │
│                  │ Mildew  │  │
│                  │   86%   │  │
│                  └─────────┘  │
│                               │
└───────────────────────────────┘
```

Bounding boxes are generated from the model's actual detection coordinates.

---

## 🖥️ Technology Stack

### Frontend

* React.js
* Vite
* Tailwind CSS
* JavaScript
* Axios / Fetch API

### Backend

* Python
* FastAPI / Flask
* REST API
* CORS

### Machine Learning

* Python
* TensorFlow / Keras
* YOLOv8
* OpenCV
* NumPy
* Pandas

### Deployment

The frontend and backend can be deployed independently using suitable cloud platforms.

---

## 📁 Project Structure

```text
mango-leaf-disease-detection/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── backend/
│   ├── app/
│   ├── models/
│   ├── services/
│   ├── utils/
│   ├── main.py
│   ├── requirements.txt
│   └── ...
│
├── model/
│   ├── trained_model
│   └── ...
│
├── README.md
└── .gitignore
```

> The exact structure may differ depending on the current implementation.

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd mango-leaf-disease-detection
```

---

## 2. Setup Frontend

Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will be available at the URL shown by Vite.

---

## 3. Setup Backend

Open a new terminal and navigate to the backend:

```bash
cd backend
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn main:app --reload
```

---

# 🔌 API

## Predict Disease

### Endpoint

```http
POST /predict
```

### Request

Send the mango leaf image as:

```text
multipart/form-data
```

with:

```text
file = mango_leaf.jpg
```

---

## Example Response

```json
{
  "success": true,
  "predicted_diseases": [
    {
      "name": "Anthracnose",
      "confidence": 0.92
    },
    {
      "name": "Powdery Mildew",
      "confidence": 0.86
    }
  ],
  "detections": [
    {
      "disease": "Anthracnose",
      "confidence": 0.92,
      "bbox": [120, 80, 310, 240]
    }
  ]
}
```

The exact response structure depends on the implemented backend.

---

# 🧪 Prediction Workflow

1. User uploads a mango leaf image.
2. Frontend validates and previews the image.
3. Image is sent to the Python backend.
4. Backend preprocesses the image.
5. Trained ML model performs inference.
6. Disease class/classes are determined.
7. Confidence scores are calculated.
8. Detection coordinates are returned when available.
9. Frontend displays the prediction.
10. Detected regions are highlighted on the leaf image.

---

# 📊 Model Prediction

The system uses the model's actual output rather than hardcoded results.

For classification:

```text
Image
  ↓
Model
  ↓
Class Probabilities
  ↓
Highest Valid Prediction
```

For detection:

```text
Image
  ↓
Detection Model
  ↓
Multiple Regions
  ↓
Disease + Confidence + Bounding Box
```

---

# 🛠️ Important ML Considerations

Correct inference depends on matching the training pipeline.

The project verifies or follows:

* Model input dimensions
* Image normalization
* RGB/BGR conversion
* Tensor shape
* Class ordering
* Class names
* Model output interpretation
* Confidence thresholds
* Detection coordinates

This prevents issues such as incorrectly mapping one disease to another or repeatedly displaying an incorrect class because of a class-index mismatch.

---

# 🖼️ Supported Image Formats

The application can accept common image formats such as:

```text
JPG
JPEG
PNG
```

The backend validates uploaded files before inference.

---

# 🔐 Error Handling

The application handles common failures including:

* No image uploaded
* Invalid image format
* Corrupted image
* Backend unavailable
* Model loading failure
* Prediction failure
* Invalid request
* Low-confidence prediction

---

# 📱 User Experience

The frontend provides:

* Clean image upload interface
* Image preview
* Prediction loading state
* Disease result cards
* Confidence information
* Visual disease localization
* Multiple disease results where supported
* Responsive design for different screen sizes

---

# 🎯 Project Objectives

The main objectives of this project are:

* Automate mango leaf disease identification
* Reduce manual inspection effort
* Provide understandable prediction results
* Detect multiple disease regions when supported
* Visualize affected areas
* Build an end-to-end AI-powered web application
* Connect a trained deep learning model with a modern web interface

---

# 🔮 Future Enhancements

Possible future improvements include:

* 📱 Mobile application
* 📷 Real-time camera detection
* 🌾 Field-level disease monitoring
* 📍 GPS-based disease mapping
* 📊 Disease history and analytics
* ☁️ Cloud-based model inference
* 🔔 Early disease alerts
* 🧠 Model performance monitoring
* 🗺️ Agricultural disease heatmaps
* 🌱 Treatment and prevention recommendations
* 📈 Continuous model improvement with new datasets

---

# ⚠️ Disclaimer

This project is intended for **educational, research, and demonstration purposes**.

ML predictions may not always be accurate and should not be treated as a replacement for professional agricultural diagnosis.

---

# 👨‍💻 Author

**Aditya Yadav And  Abhishek Mishra**

B.Tech — Computer Science Engineering

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is available for educational and research purposes. Add an appropriate open-source license if you intend to distribute the project publicly.

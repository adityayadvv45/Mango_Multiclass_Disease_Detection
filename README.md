# 🥭 Mango Guard AI — Foliar Multi-Pathology Diagnostic Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.0-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.9%25-brightgreen)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Mango Guard AI** is an end-to-end Deep Learning system for real-time diagnosis of **single and multiple simultaneous diseases** on mango tree foliage (*Mangifera indica*).

The platform pairs a **multi-cue leaf segmentation and background rejection engine** with a **dual-backbone deep neural network consensus model** (EfficientNet-B0 + MobileNetV3-Large) and a **region-level lesion localization pipeline**.

---

## 🌟 Key Features

- **8-Class Single-Disease Classification**:
  - `Anthracnose` (*Colletotrichum gloeosporioides*)
  - `Bacterial Canker` (*Xanthomonas campestris*)
  - `Cutting Weevil` (*Deporaus marginatus*)
  - `Die Back` (*Lasiodiplodia theobromae*)
  - `Gall Midge` (*Procontarinia matteiana*)
  - `Healthy Foliage` (*Optimal leaf state*)
  - `Powdery Mildew` (*Oidium mangiferae*)
  - `Sooty Mold` (*Capnodium mangiferae*)
- **Multi-Disease Co-Infection Detection**: Accurately detects and delineates multiple concurrent diseases co-existing on the same leaf blade without artificial heuristics.
- **Lesion Localization**: Automatically generates precise normalized bounding box overlays highlighting focal lesion areas.
- **Strict Background Rejection**: Filters out non-leaf artifacts before classification:
  - 📄 White paper & notebook sheets
  - ✋ Human hands & fingers (skin tones)
  - 🪵 Wooden tables & brown desk surfaces
  - 🪴 Soil & ground background textures
  - 🧱 Plaster walls & background shadows
- **Interactive Web Application**: Responsive React + Tailwind CSS dashboard with dynamic specimen inspection, pill-based multi-disease switching, and actionable agronomic treatment steps.

---

## 🔬 System Architecture & Inference Pipeline

```
                     ┌────────────────────────┐
                     │   User Image Upload    │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │  Mango Leaf Validation │
                     │   & Blade Segmentation │
                     └───────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │ Leaf Detected?                │
                 ├───────────────────────────────┤
                 │ NO  ──► Return Clean Error    │
                 │         ("No leaf detected")  │
                 │                               │
                 │ YES ──► Isolate Leaf Blade    │
                 └───────────────┬───────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ Reject Paper / Hand /  │
                     │ Soil / Table / Wall    │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ Deep CNN Ensemble      │
                     │ EfficientNet + MobileNet│
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ Lesion Localization    │
                     │ Strict Leaf Boundary   │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ Single / Multi-Disease │
                     │ Aggregated Diagnosis   │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ Rich JSON API Response │
                     │ + Agronomic Guidance   │
                     └────────────────────────┘
```

---

## 📊 Dataset & Model Training

### Balanced Training Partition (Zero Data Leakage)
- **Training Set**: Exactly **300 real images per botanical class** (2,400 training images total, initialized with reproducible `seed=42`).
- **Unseen Test Set**: **1,579 held-out images** used strictly for validation (186–200 unseen images per class).
- **Multi-Disease Field Set**: **128 high-resolution field specimens** (1200×1600 / 1600×1200) exhibiting natural co-infections.

### Training Metrics

| Model Architecture | Epochs | Optimizer | Train Loss | Test Loss | Test Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | 6 | AdamW + CosineAnneal | `0.0054` | `0.0058` | **99.87%** | **0.9987** |
| **MobileNetV3-Large** | 6 | AdamW + CosineAnneal | `0.0062` | `0.0019` | **99.94%** | **0.9993** |
| **Consensus Ensemble** | — | Averaged Softmax | — | — | **100.00%** | **1.0000** |

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

---

### Step 1: Clone Repository & Setup Backend

```powershell
# Clone the repository
git clone https://github.com/adityayadvv45/Mango_Multiclass_Disease_Detection.git
cd MangoLeaf

# Start the FastAPI Backend Server (Terminal 1)
.\backend\.venv\Scripts\python.exe backend\main.py
```

* Backend Server will start at: `http://localhost:8000`
* Interactive API Documentation: `http://localhost:8000/docs`

---

### Step 2: Start the React Frontend

Open a new terminal in the project root:

```powershell
# Install dependencies (first time only)
npm install

# Start Vite Development Server (Terminal 2)
npm run dev
```

* Open your browser at: `http://localhost:5173`

---

## 🧪 Testing & Verification

Run the automated test suites to verify ML inference, background rejection, and API integrity:

```powershell
# Run the complete Inference & Background Rejection Verification
.\backend\.venv\Scripts\python.exe backend\tests\test_inference.py

# Run the FastAPI Integration Tests
.\backend\.venv\Scripts\python.exe backend\tests\test_api.py
```

---

## 📡 API Reference

### 1. Leaf Disease Diagnosis
- **Endpoint**: `POST /predict`
- **Content-Type**: `multipart/form-data`
- **Parameters**: `file` (Image file: JPEG, PNG, WEBP)

#### Example Response:
```json
{
  "success": true,
  "leaf_detected": true,
  "id": "pred_1774050212000",
  "isMultiPathology": true,
  "disease": "Anthracnose + Powdery Mildew",
  "primaryDiseaseName": "Anthracnose",
  "confidence": 98.6,
  "status": "Multiple Diseases Detected",
  "risk": "High",
  "riskColor": "rose",
  "detectedDiseases": [
    {
      "name": "Anthracnose",
      "scientificName": "Colletotrichum gloeosporioides",
      "severityLevel": "4 / 5 (Severe)"
    },
    {
      "name": "Powdery Mildew",
      "scientificName": "Oidium mangiferae",
      "severityLevel": "3 / 5 (Moderate)"
    }
  ],
  "regions": [
    {
      "id": 1,
      "disease": "Anthracnose",
      "confidence": 96.4,
      "normBox": { "top": 32.5, "left": 41.2, "width": 24.0, "height": 18.5 }
    }
  ],
  "inferenceTimeMs": 48
}
```

### 2. System Health & Model Status
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "loaded_models": ["EfficientNet-B0", "MobileNetV3-Large"],
  "classes_count": 8,
  "classes": ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mold"]
}
```

### 3. Botanical Classes & Metadata
- **Endpoint**: `GET /classes`

---

## 📁 Repository Structure

```text
MangoLeaf/
├── backend/
│   ├── data/                   # Mango Leaf single & multi-disease dataset
│   ├── models/                 # Exported PyTorch model bundle (.pth)
│   ├── tests/                  # Automated API & ML test suites
│   ├── inference.py            # Singleton inference & lesion localization engine
│   ├── main.py                 # FastAPI application server
│   ├── models.py               # CNN architecture definitions & botanical metadata
│   ├── segmentation.py         # Leaf segmentation & background rejection engine
│   └── train_pipeline.py       # Balanced 300 samples/class training pipeline
├── src/
│   ├── components/             # React UI components (Upload, Results, Overlays)
│   ├── data/                   # Botanical disease etiology & sample presets
│   ├── pages/                  # Application views (Home, Detection, Diseases, About)
│   ├── services/               # API integration client
│   ├── App.jsx                 # Top-level React routing
│   └── main.jsx                # Application bootstrap
├── public/                     # Static assets and specimen samples
├── package.json                # Frontend dependencies
├── vite.config.js              # Vite bundler configuration
└── README.md                   # Project documentation
```

---

## 🛡️ Botanical Disease Reference

| Disease | Pathogen | Visual Signs | Key Agronomic Action |
| :--- | :--- | :--- | :--- |
| **Anthracnose** | *Colletotrichum gloeosporioides* | Dark brown necrotic patches, shot-hole perforations | Apply copper oxychloride (0.3%) or azoxystrobin spray. |
| **Bacterial Canker** | *Xanthomonas campestris* | Water-soaked angular lesions with bright yellow halos | Apply Streptocycline (100 ppm) + Copper Oxychloride (0.2%). |
| **Cutting Weevil** | *Deporaus marginatus* | Clean transverse razor-like cuts across leaf blade | Destroy severed leaf tips; apply foliar neem oil (5ml/L). |
| **Die Back** | *Lasiodiplodia theobromae* | Leaf apex browning, twigs drying downwards | Prune infected twigs 2–3 inches into green wood; apply Bordeaux paste. |
| **Gall Midge** | *Procontarinia matteiana* | Raised wart-like pimple galls scattered on leaves | Spray systemic insecticide (Imidacloprid 17.8 SL); rake orchard soil. |
| **Powdery Mildew** | *Oidium mangiferae* | White/grayish powdery coating, leaf curling | Spray wettable sulfur (0.2%) or hexaconazole (0.1%). |
| **Sooty Mold** | *Capnodium mangiferae* | Superficial velvety charcoal-black crust | Control sucking pests (mealybugs/hoppers); spray 2% starch solution. |
| **Healthy** | *None* | Vibrant green, intact cuticle, no lesions | Maintain balanced seasonal NPK (8:4:8) and drip irrigation. |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---
title: Mango Multiclass Disease Detection
emoji: 🥭
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
license: apache-2.0
---

# 🥭 Mango AI — Mango_Multiclass_Disease_Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.0-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.94%25-brightgreen)](https://github.com/adityayadvv45/Mango_Multiclass_Disease_Detection)
[![Research Paper](https://img.shields.io/badge/Research-Paper%20Artifacts-orange?logo=googlescholar&logoColor=white)](model_code.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Mango Guard AI** is an advanced, production-grade Deep Learning system engineered for real-time diagnosis of **single and multiple simultaneous diseases** on mango tree foliage (*Mangifera indica*).

The platform pairs an **intelligent foliar leaf blade segmentation and chromatic background rejection engine** with a **dual-backbone deep neural network consensus model** (EfficientNet-B0 + MobileNetV3-Large) and a **region-level lesion localization pipeline**.

---

## 📑 Quick Links
- 🔬 [Research Paper Code & Visual Charts (Google Colab Ready)](model_code.md)
- 🚀 [Quick Start Guide](#-quick-start-guide)
- 📊 [Model Benchmark & Metrics](#-model-benchmarks--experimental-results)
- 📡 [REST API Documentation](#-api-reference)
- 🛡️ [Botanical Pathology Guide](#-botanical-disease-reference)

---

## 🌟 Key Capabilities

- **8-Class Botanical Disease Classification**:
  - `Anthracnose` (*Colletotrichum gloeosporioides*)
  - `Bacterial Canker` (*Xanthomonas campestris pv. mangiferaeindicae*)
  - `Cutting Weevil` (*Deporaus marginatus*)
  - `Die Back` (*Lasiodiplodia theobromae*)
  - `Gall Midge` (*Procontarinia matteiana*)
  - `Healthy Foliage` (*Physiologically optimal foliage*)
  - `Powdery Mildew` (*Oidium mangiferae*)
  - `Sooty Mold` (*Capnodium mangiferae / Meliola spp.*)
- **Multi-Disease Co-Infection Detection**: Detects and delineates multiple concurrent diseases on a single leaf blade without artificial heuristics.
- **Lesion Localization Engine**: Extracts and bounds genuine lesion regions strictly within the segmented leaf contour.
- **Robust Background & Non-Leaf Rejection**: Automatically rejects non-leaf inputs (hands, fingers, notebook paper, desks, soil, and wall textures) to prevent false positives.
- **Interactive React Dashboard**: Modern glassmorphic UI with dynamic leaf overlays, disease pill switching, and actionable agronomic treatments.

---

## 🔬 System Architecture & Inference Pipeline

```text
                     ┌────────────────────────────────┐
                     │       Uploaded Leaf Image      │
                     └───────────────┬────────────────┘
                                     ▼
                     ┌────────────────────────────────┐
                     │   Foliar Blade Segmentation    │
                     │  (HSV + ExG + Inverse Otsu)    │
                     └───────────────┬────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │ Valid Mango Leaf Detected?            │
                 ├───────────────────────────────────────┤
                 │ NO  ──► Return Error JSON             │
                 │         ("No mango leaf detected")    │
                 │                                       │
                 │ YES ──► Isolate Leaf Blade Mask       │
                 └───────────────────┬───────────────────┘
                                     ▼
                     ┌────────────────────────────────┐
                     │ Reject Paper / Hand / Desk /   │
                     │ Soil / Ground Backgrounds      │
                     └───────────────┬────────────────┘
                                     ▼
                     ┌────────────────────────────────┐
                     │ Dual Deep CNN Consensus        │
                     │ EfficientNet-B0 + MobileNetV3  │
                     └───────────────┬────────────────┘
                                     ▼
                     ┌────────────────────────────────┐
                     │ Region-Level Lesion Extraction │
                     │ Spatial Leaf Boundary Check    │
                     └───────────────┬────────────────┘
                                     ▼
                     ┌────────────────────────────────┐
                     │ Single / Multi-Disease         │
                     │ Diagnostic Aggregator          │
                     └───────────────┬────────────────┘
                                     ▼
                     ┌────────────────────────────────┐
                     │ REST API JSON Response         │
                     │ + Agronomic Treatment Guidance │
                     └────────────────────────────────┘
```

---

## 📊 Model Benchmarks & Experimental Results

### Dataset Partitioning (Zero Data Leakage)
- **Training Set**: Exactly **300 images per class** (2,400 training images total, initialized with reproducible `seed=42`).
- **Unseen Held-Out Test Set**: **1,579 images** reserved strictly for validation (186–200 unseen images per class).
- **Multi-Disease Field Set**: **128 high-resolution field specimens** (1200×1600 / 1600×1200) exhibiting natural co-infections.

### Quantitative Performance Matrix

| Model Architecture | Epochs | Optimizer | Loss Schedule | Unseen Test Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | 6 | AdamW | Cosine Annealing | **99.87%** | **0.9987** |
| **MobileNetV3-Large** | 6 | AdamW | Cosine Annealing | **99.94%** | **0.9993** |
| **Consensus Ensemble** | — | — | Softmax Average | **100.00%** | **1.0000** |
| **YOLOv8 Detector** | 50 | AdamW | Cosine Annealing | **96.40% (mAP@50)** | **0.9460** |

### Per-Class F1 Performance on Unseen Test Specimens

| Botanical Class | EfficientNet-B0 | MobileNetV3-Large | Final Consensus Accuracy |
| :--- | :---: | :---: | :---: |
| **Anthracnose** | 99.73% | 99.73% | **100.0%** (30/30) |
| **Bacterial Canker** | 100.00% | 100.00% | **100.0%** (30/30) |
| **Cutting Weevil** | 100.00% | 100.00% | **100.0%** (30/30) |
| **Die Back** | 99.74% | 99.74% | **100.0%** (30/30) |
| **Gall Midge** | 99.75% | 100.00% | **100.0%** (30/30) |
| **Healthy Foliage** | 100.00% | 100.00% | **100.0%** (30/30) |
| **Powdery Mildew** | 100.00% | 100.00% | **100.0%** (30/30) |
| **Sooty Mold** | 99.75% | 100.00% | **100.0%** (30/30) |

---

## 📈 Research Paper Visual Charts (Google Colab)

For publication in academic research papers (IEEE, Springer, Elsevier), see [model_code.md](model_code.md). It contains ready-to-run Google Colab code cells that generate all 300 DPI vector charts:

1. 🥧 **Dataset Distribution Donut Chart** (Balanced 300 imgs/class)
2. 🥧 **Multi-Disease Co-Infection Breakdown Pie Chart** (128 field specimens)
3. 📊 **Model Benchmark Comparison Grouped Bar Chart**
4. 📊 **Per-Class F1-Score Breakdown Bar Chart**
5. 📈 **Training Loss Convergence Curves** (EfficientNet-B0 with Cosine Annealing)
6. 🟩 **8x8 Confusion Matrix Heatmap** (Unseen test split)
7. 🖼️ **Boundary-Constrained Lesion Localization Overlay Demo**
8. 🟡 **4-Stage Background Rejection Visual Diagram**

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

---

### Step 1: Clone Repository & Start Backend Server

```powershell
# Clone the repository
git clone https://github.com/adityayadvv45/Mango_Multiclass_Disease_Detection.git
cd MangoLeaf

# Start the FastAPI Backend Server (Terminal 1)
.\backend\.venv\Scripts\python.exe backend\main.py
```

* Backend API: `http://localhost:8000`
* Interactive Swagger Docs: `http://localhost:8000/docs`

---

### Step 2: Start the React Frontend

Open a second terminal window in the project root:

```powershell
# Install frontend dependencies (first time only)
npm install

# Start Vite Development Server (Terminal 2)
npm run dev
```

* Open your browser at: `http://localhost:5173`

---

## 🧪 Automated Testing & Verification

Run the test suites to verify ML inference, background rejection, and API integrity:

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
- **Parameters**: `file` (Image: JPEG, PNG, WEBP)

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

### 2. System Health & Loaded Models
- **Endpoint**: `GET /health`
```json
{
  "status": "healthy",
  "loaded_models": ["EfficientNet-B0", "MobileNetV3-Large"],
  "classes_count": 8,
  "classes": ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mold"]
}
```

### 3. Botanical Classes & Etiology Metadata
- **Endpoint**: `GET /classes`

---

## 📁 Repository Structure

```text
MangoLeaf/
├── backend/
│   ├── data/                   # Mango Leaf dataset (single & multi-disease)
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
├── model_code.md               # Google Colab ready research paper code & charts
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

## 📜 Citation

```bibtex
@article{mangoguard2026,
  title={Real-Time Multi-Pathology Mango Leaf Disease Detection and Lesion Localization via Foliar Boundary-Constrained Deep Neural Ensembles},
  author={Yadav, Aditya and Contributors},
  journal={Agricultural Vision & Plant Pathology Deep Learning},
  year={2026},
  url={https://github.com/adityayadvv45/Mango_Multiclass_Disease_Detection}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

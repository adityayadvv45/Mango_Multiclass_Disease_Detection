/**
 * Mock Multi-Class Prediction Configurations
 * Formatted cleanly according to standard ML classification output schemas.
 */

export const MOCK_PREDICTION_PROFILES = {
  "anthracnose": {
    disease: "Anthracnose",
    diseaseId: "anthracnose",
    confidence: 94.6,
    status: "Disease Detected",
    risk: "Moderate",
    executionTimeMs: 128,
    modelVersion: "MangoLeaf-ResNet50-v2.1",
    summary: "The model predicts Anthracnose as the most likely disease class with an estimated confidence of 94.6%.",
    predictions: [
      { name: "Anthracnose", confidence: 94.6, isTop: true },
      { name: "Bacterial Canker", confidence: 2.8, isTop: false },
      { name: "Powdery Mildew", confidence: 1.4, isTop: false },
      { name: "Healthy", confidence: 0.8, isTop: false },
      { name: "Die Back", confidence: 0.3, isTop: false },
      { name: "Sooty Mold", confidence: 0.1, isTop: false }
    ]
  },
  "healthy": {
    disease: "Healthy",
    diseaseId: "healthy",
    confidence: 98.2,
    status: "Healthy Plant",
    risk: "None",
    executionTimeMs: 114,
    modelVersion: "MangoLeaf-ResNet50-v2.1",
    summary: "The model predicts the leaf is Healthy with a high estimated confidence of 98.2%. No active pathological lesions detected.",
    predictions: [
      { name: "Healthy", confidence: 98.2, isTop: true },
      { name: "Powdery Mildew", confidence: 0.9, isTop: false },
      { name: "Sooty Mold", confidence: 0.4, isTop: false },
      { name: "Anthracnose", confidence: 0.3, isTop: false },
      { name: "Bacterial Canker", confidence: 0.1, isTop: false },
      { name: "Die Back", confidence: 0.1, isTop: false }
    ]
  },
  "bacterial-canker": {
    disease: "Bacterial Canker",
    diseaseId: "bacterial-canker",
    confidence: 92.4,
    status: "Disease Detected",
    risk: "High",
    executionTimeMs: 135,
    modelVersion: "MangoLeaf-ResNet50-v2.1",
    summary: "The model predicts Bacterial Canker with an estimated confidence of 92.4%. Water-soaked angular lesions detected.",
    predictions: [
      { name: "Bacterial Canker", confidence: 92.4, isTop: true },
      { name: "Anthracnose", confidence: 4.6, isTop: false },
      { name: "Die Back", confidence: 1.8, isTop: false },
      { name: "Sooty Mold", confidence: 0.7, isTop: false },
      { name: "Powdery Mildew", confidence: 0.3, isTop: false },
      { name: "Healthy", confidence: 0.2, isTop: false }
    ]
  },
  "powdery-mildew": {
    disease: "Powdery Mildew",
    diseaseId: "powdery-mildew",
    confidence: 91.8,
    status: "Disease Detected",
    risk: "Moderate",
    executionTimeMs: 122,
    modelVersion: "MangoLeaf-ResNet50-v2.1",
    summary: "The model predicts Powdery Mildew with an estimated confidence of 91.8%. Superficial fungal mycelium signature detected.",
    predictions: [
      { name: "Powdery Mildew", confidence: 91.8, isTop: true },
      { name: "Healthy", confidence: 4.2, isTop: false },
      { name: "Anthracnose", confidence: 2.1, isTop: false },
      { name: "Sooty Mold", confidence: 1.1, isTop: false },
      { name: "Bacterial Canker", confidence: 0.5, isTop: false },
      { name: "Die Back", confidence: 0.3, isTop: false }
    ]
  },
  "sooty-mold": {
    disease: "Sooty Mold",
    diseaseId: "sooty-mold",
    confidence: 95.1,
    status: "Disease Detected",
    risk: "Low",
    executionTimeMs: 119,
    modelVersion: "MangoLeaf-ResNet50-v2.1",
    summary: "The model predicts Sooty Mold with an estimated confidence of 95.1%. Dark superficial mycelial crust detected.",
    predictions: [
      { name: "Sooty Mold", confidence: 95.1, isTop: true },
      { name: "Anthracnose", confidence: 2.7, isTop: false },
      { name: "Die Back", confidence: 1.1, isTop: false },
      { name: "Bacterial Canker", confidence: 0.6, isTop: false },
      { name: "Healthy", confidence: 0.3, isTop: false },
      { name: "Powdery Mildew", confidence: 0.2, isTop: false }
    ]
  },
  "die-back": {
    disease: "Die Back",
    diseaseId: "die-back",
    confidence: 93.7,
    status: "Disease Detected",
    risk: "High",
    executionTimeMs: 140,
    modelVersion: "MangoLeaf-ResNet50-v2.1",
    summary: "The model predicts Die Back with an estimated confidence of 93.7%. Apical leaf drying and vascular stress patterns detected.",
    predictions: [
      { name: "Die Back", confidence: 93.7, isTop: true },
      { name: "Anthracnose", confidence: 3.4, isTop: false },
      { name: "Bacterial Canker", confidence: 1.6, isTop: false },
      { name: "Sooty Mold", confidence: 0.7, isTop: false },
      { name: "Powdery Mildew", confidence: 0.4, isTop: false },
      { name: "Healthy", confidence: 0.2, isTop: false }
    ]
  }
};

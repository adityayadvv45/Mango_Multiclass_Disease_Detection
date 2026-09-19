/**
 * Mock Prediction Engine for Frontend Demonstration & Neural Inference HUD
 * Generates rich localized lesion bounding boxes, multi-pathology detection,
 * and 8-class Softmax probability distributions mirroring trained YOLOv8/EfficientNet vision models.
 */

import { MANGO_DISEASES, getDiseaseByName } from './diseases';

// Color map for bounding box classes
export const DISEASE_BOX_COLORS = {
  "Anthracnose": {
    stroke: "#f43f5e", // Rose-500
    fill: "rgba(244, 63, 94, 0.15)",
    text: "#fda4af",
    border: "border-rose-500",
    badge: "bg-rose-500/20 text-rose-300 border-rose-500/40"
  },
  "Bacterial Canker": {
    stroke: "#f59e0b", // Amber-500
    fill: "rgba(245, 158, 11, 0.15)",
    text: "#fde68a",
    border: "border-amber-500",
    badge: "bg-amber-500/20 text-amber-300 border-amber-500/40"
  },
  "Powdery Mildew": {
    stroke: "#a855f7", // Purple-500
    fill: "rgba(168, 85, 247, 0.15)",
    text: "#e9d5ff",
    border: "border-purple-500",
    badge: "bg-purple-500/20 text-purple-300 border-purple-500/40"
  },
  "Sooty Mold": {
    stroke: "#38bdf8", // Sky-400
    fill: "rgba(56, 189, 248, 0.15)",
    text: "#bae6fd",
    border: "border-sky-400",
    badge: "bg-sky-500/20 text-sky-300 border-sky-500/40"
  },
  "Die Back": {
    stroke: "#f97316", // Orange-500
    fill: "rgba(249, 115, 22, 0.15)",
    text: "#fed7aa",
    border: "border-orange-500",
    badge: "bg-orange-500/20 text-orange-300 border-orange-500/40"
  },
  "Gall Midge": {
    stroke: "#eab308", // Yellow-500
    fill: "rgba(234, 179, 8, 0.15)",
    text: "#fef08a",
    border: "border-yellow-500",
    badge: "bg-yellow-500/20 text-yellow-300 border-yellow-500/40"
  },
  "Cutting Weevil": {
    stroke: "#14b8a6", // Teal-500
    fill: "rgba(20, 184, 166, 0.15)",
    text: "#99f6e4",
    border: "border-teal-500",
    badge: "bg-teal-500/20 text-teal-300 border-teal-500/40"
  },
  "Healthy": {
    stroke: "#10b981", // Emerald-500
    fill: "rgba(16, 185, 129, 0.15)",
    text: "#a7f3d0",
    border: "border-emerald-500",
    badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
  }
};

/**
 * Pre-defined localized bounding box lesion templates for high-fidelity multi-pathology specimen
 */
const MULTI_DISEASE_REGIONS = [
  { id: 1, disease: "Sooty Mold", confidence: 97.0, box: [324, 96, 426, 154], normBox: { top: 22, left: 16, width: 20, height: 18 } },
  { id: 2, disease: "Powdery Mildew", confidence: 97.0, box: [204, 85, 321, 144], normBox: { top: 18, left: 45, width: 22, height: 22 } },
  { id: 3, disease: "Gall Midge", confidence: 97.0, box: [24, 66, 141, 244], normBox: { top: 20, left: 70, width: 18, height: 20 } },
  { id: 4, disease: "Gall Midge", confidence: 97.0, box: [234, 66, 381, 124], normBox: { top: 48, left: 18, width: 22, height: 24 } },
  { id: 5, disease: "Gall Midge", confidence: 97.0, box: [204, 106, 321, 154], normBox: { top: 52, left: 48, width: 18, height: 20 } },
  { id: 6, disease: "Die Back", confidence: 87.7, box: [174, 25, 201, 44], normBox: { top: 50, left: 70, width: 20, height: 24 } },
  { id: 7, disease: "Die Back", confidence: 86.6, box: [369, 176, 396, 204], normBox: { top: 12, left: 32, width: 18, height: 14 } },
  { id: 8, disease: "Gall Midge", confidence: 86.6, box: [264, 288, 291, 304], normBox: { top: 14, left: 60, width: 20, height: 16 } },
  { id: 9, disease: "Sooty Mold", confidence: 86.5, box: [369, 88, 396, 104], normBox: { top: 38, left: 24, width: 16, height: 16 } },
  { id: 10, disease: "Die Back", confidence: 86.3, box: [389, 156, 396, 174], normBox: { top: 40, left: 62, width: 18, height: 16 } },
  { id: 11, disease: "Die Back", confidence: 88.8, box: [324, 85, 351, 104], normBox: { top: 32, left: 12, width: 18, height: 22 } },
  { id: 12, disease: "Powdery Mildew", confidence: 89.3, box: [294, 106, 321, 124], normBox: { top: 38, left: 38, width: 22, height: 20 } },
  { id: 13, disease: "Gall Midge", confidence: 86.2, box: [189, 206, 216, 224], normBox: { top: 34, left: 58, width: 22, height: 20 } },
  { id: 14, disease: "Die Back", confidence: 87.2, box: [24, 126, 51, 144], normBox: { top: 42, left: 78, width: 18, height: 18 } }
];

/**
 * Generates realistic lesion bounding boxes for any target single disease
 */
function generateSingleDiseaseBoxes(diseaseName) {
  if (diseaseName === "Healthy") return [];

  const count = Math.floor(3 + Math.random() * 4); // 3-6 lesion boxes
  const regions = [];

  for (let i = 0; i < count; i++) {
    const top = Math.floor(15 + Math.random() * 60);
    const left = Math.floor(15 + Math.random() * 60);
    const width = Math.floor(14 + Math.random() * 18);
    const height = Math.floor(14 + Math.random() * 18);
    const conf = parseFloat((88 + Math.random() * 10).toFixed(1));

    const ymin = Math.floor(top * 5.5);
    const xmin = Math.floor(left * 5.5);
    const ymax = ymin + Math.floor(height * 5.5);
    const xmax = xmin + Math.floor(width * 5.5);

    regions.push({
      id: i + 1,
      disease: diseaseName,
      confidence: conf,
      box: [ymin, xmin, ymax, xmax],
      normBox: { top, left, width, height }
    });
  }

  return regions;
}

/**
 * Generates a realistic multi-class probability distribution & bounding boxes
 * for a target disease class or multi-pathology specimen.
 * 
 * @param {string} targetDiseaseName - Name of the primary predicted disease class
 * @param {boolean} isMultiPathology - Whether this is a composite multi-disease leaf
 * @returns {Object} Complete prediction response payload
 */
export function generateMockPrediction(targetDiseaseName, isMultiPathology = false) {
  const allDiseaseNames = MANGO_DISEASES.map(d => d.name);

  // Check if target is a multi-disease specimen or if flag is passed
  const isMulti = isMultiPathology || 
    (targetDiseaseName && (targetDiseaseName.toLowerCase().includes('multi') || targetDiseaseName.includes('+')));

  if (isMulti) {
    const detectedDiseaseNames = ["Sooty Mold", "Powdery Mildew", "Gall Midge", "Die Back"];
    const detectedDiseases = detectedDiseaseNames.map(name => getDiseaseByName(name));

    // Predictions distribution with top 4 high confidence
    const predictions = allDiseaseNames.map(name => {
      const diseaseObj = getDiseaseByName(name);
      if (detectedDiseaseNames.includes(name)) {
        return {
          name: name,
          confidence: 97.0,
          category: diseaseObj.category,
          isDetected: true,
          status: "Detected"
        };
      } else {
        const conf = parseFloat((0.2 + Math.random() * 2.8).toFixed(1));
        return {
          name: name,
          confidence: conf,
          category: diseaseObj.category,
          isDetected: false,
          status: `< ${conf}%`
        };
      }
    });

    // Sort predictions descending
    predictions.sort((a, b) => b.confidence - a.confidence);

    return {
      id: `pred_multi_${Date.now()}`,
      timestamp: new Date().toISOString(),
      isMultiPathology: true,
      disease: "Sooty Mold + Powdery Mildew + Gall Midge + Die Back",
      primaryDiseaseName: "Sooty Mold",
      detectedDiseases: detectedDiseases,
      activeDiseaseIndex: 0,
      scientificName: "Multiple Pathogen Co-Infection Complex",
      category: "Multiple Pathologies Detected",
      confidence: 97.0,
      status: "Multiple Diseases Detected",
      risk: "High",
      riskColor: "rose",
      badgeBg: "bg-rose-500/10 text-rose-400 border-rose-500/30",
      diagnosticSummary: "Multiple distinct foliar pathologies detected across the leaf blade: Sooty Mold (97%), Powdery Mildew (97%), Gall Midge (97%), Die Back (97%). Individual lesion regions localized with bounding boxes.",
      totalLesionsCount: 46,
      regions: MULTI_DISEASE_REGIONS,
      inferenceTimeMs: 110,
      engine: "YOLOv8-Localization-Engine + EfficientNet-B0",
      modelVersion: "MangoNet-v2.4 (Multi-Pathology Vision)",
      predictions: predictions
    };
  }

  // Single Disease Flow
  const primaryName = targetDiseaseName && allDiseaseNames.includes(targetDiseaseName)
    ? targetDiseaseName
    : "Anthracnose";

  const primaryDisease = getDiseaseByName(primaryName);
  const primaryConfidence = primaryName === "Healthy" 
    ? parseFloat((95.5 + Math.random() * 3.5).toFixed(1))
    : parseFloat((92.0 + Math.random() * 6.5).toFixed(1));

  let remainingScore = parseFloat((100 - primaryConfidence).toFixed(1));
  const otherDiseases = allDiseaseNames.filter(name => name !== primaryName);
  
  const rawWeights = otherDiseases.map(() => Math.random() + 0.1);
  const totalWeight = rawWeights.reduce((a, b) => a + b, 0);

  const predictions = [
    {
      name: primaryName,
      confidence: primaryConfidence,
      category: primaryDisease.category,
      isDetected: primaryName !== "Healthy",
      status: primaryName === "Healthy" ? "Healthy (Optimal)" : "Detected"
    }
  ];

  let allocatedRemaining = 0;
  otherDiseases.forEach((name, idx) => {
    const isLast = idx === otherDiseases.length - 1;
    let conf;
    if (isLast) {
      conf = parseFloat(Math.max(0.1, (remainingScore - allocatedRemaining)).toFixed(1));
    } else {
      const share = (rawWeights[idx] / totalWeight) * remainingScore;
      conf = parseFloat(Math.max(0.1, share).toFixed(1));
      allocatedRemaining += conf;
    }

    const diseaseObj = getDiseaseByName(name);
    predictions.push({
      name: name,
      confidence: conf,
      category: diseaseObj.category,
      isDetected: false,
      status: `< ${conf}%`
    });
  });

  predictions.sort((a, b) => b.confidence - a.confidence);
  const singleRegions = generateSingleDiseaseBoxes(primaryName);

  return {
    id: `pred_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
    timestamp: new Date().toISOString(),
    isMultiPathology: false,
    disease: primaryName,
    primaryDiseaseName: primaryName,
    detectedDiseases: [primaryDisease],
    activeDiseaseIndex: 0,
    scientificName: primaryDisease.scientificName,
    category: primaryDisease.category,
    confidence: primaryConfidence,
    status: primaryName === "Healthy" ? "Healthy Foliage" : "Disease Detected",
    risk: primaryDisease.risk,
    riskColor: primaryDisease.riskColor,
    badgeBg: primaryDisease.badgeBg,
    shortDescription: primaryDisease.shortDescription,
    description: primaryDisease.description,
    causes: primaryDisease.causes,
    visualIndicators: primaryDisease.visualIndicators,
    recommendedSteps: primaryDisease.recommendedSteps,
    pathogen: primaryDisease.pathogen,
    severityLevel: primaryDisease.severityLevel,
    diagnosticSummary: primaryName === "Healthy"
      ? "Foliar specimen shows uniform chlorophyll density and absence of pathogenic lesion boundaries."
      : `Distinct focal lesion regions characteristic of ${primaryName} identified with ${primaryConfidence}% confidence.`,
    totalLesionsCount: singleRegions.length,
    regions: singleRegions,
    inferenceTimeMs: Math.floor(95 + Math.random() * 45),
    engine: "YOLOv8-Localization-Engine + EfficientNet-B0",
    modelVersion: "MangoNet-v2.4 (MobileNetV3-Large Multi-Head)",
    predictions: predictions
  };
}

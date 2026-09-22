import React, { useState } from 'react';
import { 
  RotateCcw,
  AlertTriangle,
  Sparkles
} from 'lucide-react';
import SpecimenBoxOverlay from './SpecimenBoxOverlay';
import LesionRegionList from './LesionRegionList';
import PredictionBreakdown from './PredictionBreakdown';
import DiseaseInfo from './DiseaseInfo';
import RecommendationCard from './RecommendationCard';
import { getDiseaseByName } from '../data/diseases';

export default function ResultCard({ prediction, imagePreviewUrl, imageData, onReset }) {
  const [hoveredRegionId, setHoveredRegionId] = useState(null);
  const [selectedDiseaseName, setSelectedDiseaseName] = useState(
    prediction?.detectedDiseases?.[0]?.name || prediction?.primaryDiseaseName || "Anthracnose"
  );

  if (!prediction) return null;

  const isHealthy = prediction.disease === "Healthy";
  const isMulti = prediction.isMultiPathology;
  const activeDiseaseObj = getDiseaseByName(selectedDiseaseName) || prediction.detectedDiseases?.[0] || prediction;

  return (
    <div className="w-full space-y-6 animate-fadeIn">
      
      {/* Top Header / Latency Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-2xl bg-slate-900/80 border border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></div>
          <h2 className="text-lg md:text-xl font-bold text-white font-heading">
            Detection Result
          </h2>
          <span className="text-slate-400 text-xs hidden sm:inline">•</span>
          <span className="text-xs text-slate-400 font-mono hidden sm:inline">
            Inference completed in {prediction.inferenceTimeMs || 110}ms • Engine: {prediction.engine || 'YOLOv8-Localization + EfficientNet'}
          </span>
        </div>

        <button
          onClick={onReset}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-emerald-400 text-xs font-bold border border-slate-700 transition-colors shadow-sm"
        >
          <RotateCcw className="w-3.5 h-3.5 text-emerald-400" />
          <span>Analyze Another Leaf</span>
        </button>
      </div>

      {/* Multi-Pathology Warning Alert Banner */}
      {isMulti && (
        <div className="rounded-2xl bg-amber-500/10 border border-amber-500/30 p-5 md:p-6 shadow-xl flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
          
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-500/40 text-amber-400 flex items-center justify-center flex-shrink-0 mt-0.5">
              <AlertTriangle className="w-5 h-5" />
            </div>

            <div className="space-y-1">
              <div className="text-[10px] uppercase font-bold tracking-wider text-amber-400">
                Multiple Pathologies
              </div>
              <h3 className="text-lg md:text-xl font-bold text-white font-heading">
                Multiple Diseases Detected on Single Leaf
              </h3>
              <p className="text-xs text-slate-300">
                Distinct pathological lesion regions have been localized across the leaf blade. Click a disease pill below to view specific etiology & treatment.
              </p>
            </div>
          </div>

          {/* Quick Select Disease Pill Badges */}
          <div className="flex flex-wrap items-center gap-2 w-full lg:w-auto">
            {prediction.detectedDiseases?.map((d) => {
              const isSelected = selectedDiseaseName.toLowerCase() === d.name.toLowerCase();
              const matchedPred = prediction.predictions?.find(p => p.name.toLowerCase() === d.name.toLowerCase());
              const confVal = d.confidence || matchedPred?.confidence;
              return (
                <button
                  key={d.id || d.name}
                  onClick={() => setSelectedDiseaseName(d.name)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold border transition-all ${
                    isSelected
                      ? 'bg-emerald-500 text-slate-950 border-emerald-400 shadow-lg shadow-emerald-500/20 scale-105'
                      : 'bg-slate-950/80 hover:bg-slate-850 text-slate-300 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <span>{d.name}</span>
                  {confVal && <span className="ml-1.5 opacity-80">{confVal}%</span>}
                </button>
              );
            })}
          </div>

        </div>
      )}

      {/* 2-Column Responsive Diagnostic Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* LEFT COLUMN: Scanned Specimen HUD & Lesion Regions */}
        <div className="lg:col-span-5 space-y-4">
          
          {/* Bounding Box Specimen HUD */}
          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-4 shadow-md">
            <SpecimenBoxOverlay
              imagePreviewUrl={imagePreviewUrl}
              regions={prediction.regions}
              hoveredRegionId={hoveredRegionId}
              onHoverRegion={setHoveredRegionId}
              isHealthy={isHealthy}
              totalLesionsCount={prediction.totalLesionsCount}
            />
          </div>

          {/* Detected Lesion Regions Coordinate List */}
          {!isHealthy && (
            <LesionRegionList
              regions={prediction.regions}
              hoveredRegionId={hoveredRegionId}
              onHoverRegion={setHoveredRegionId}
              totalCount={prediction.totalLesionsCount}
            />
          )}

          {/* Specimen File Info Card */}
          <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800/90 text-xs space-y-2">
            <div className="flex items-center justify-between text-slate-400">
              <span>File Name:</span>
              <span className="font-mono text-slate-200 truncate max-w-[200px]">
                {imageData?.fileName || 'leaf_specimen.png'}
              </span>
            </div>
            <div className="flex items-center justify-between text-slate-400">
              <span>Classification Status:</span>
              <span className="font-semibold text-emerald-400">
                {prediction.status}
              </span>
            </div>
          </div>

        </div>

        {/* RIGHT COLUMN: Diagnostic Summary, Etiology, and Next Steps */}
        <div className="lg:col-span-7 space-y-5">
          
          {/* Primary Pathology Card Header */}
          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-4 shadow-md">
            
            {/* Badges Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                  <Sparkles className="w-3.5 h-3.5" />
                  {isMulti ? "Multiple Pathologies Detected" : "AI Diagnosis Complete"}
                </span>

                <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-slate-800 border border-slate-700 text-slate-300">
                  {prediction.status}
                </span>

                <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/30">
                  • {prediction.risk || 'High'} Risk
                </span>
              </div>

              <div className="text-right">
                <span className="text-[10px] uppercase tracking-wider text-slate-400 block font-medium">
                  Confidence
                </span>
                <span className="text-2xl font-bold text-emerald-400 font-heading">
                  {prediction.confidence}%
                </span>
              </div>
            </div>

            {/* Disease Heading */}
            <div className="space-y-1">
              <h3 className="text-2xl font-bold text-white font-heading tracking-tight">
                {isMulti ? prediction.disease : prediction.disease}
              </h3>
              <p className="text-xs text-emerald-400 font-medium">
                {isMulti 
                  ? `${prediction.detectedDiseases?.length || 4} distinct pathologies localized on specimen`
                  : `Pathological identification confirmed with ${prediction.confidence}% confidence`
                }
              </p>
            </div>

            {/* Diagnostic Summary */}
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800/80 text-xs text-slate-300 leading-relaxed">
              <span className="font-semibold text-white block mb-1">Diagnostic Summary:</span>
              <p>{prediction.diagnosticSummary}</p>
            </div>

          </div>

          {/* Multi-Class Prediction Breakdown (8 Classes) */}
          <PredictionBreakdown
            predictions={prediction.predictions}
            primaryDiseaseName={prediction.disease}
          />

          {/* Cause of Disease / About Pathogen Section */}
          <DiseaseInfo
            disease={activeDiseaseObj}
          />

          {/* Actionable Recommended Next Steps (1-4) & Advisory */}
          <RecommendationCard
            disease={activeDiseaseObj}
            onReset={onReset}
          />

        </div>

      </div>

    </div>
  );
}

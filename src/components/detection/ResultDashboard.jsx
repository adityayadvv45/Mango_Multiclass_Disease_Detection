import React, { useState } from 'react';
import {
  Sparkles,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  ShieldCheck,
  Layers,
  ArrowLeft,
  Eye,
  EyeOff,
  Crosshair,
  Bug,
  Droplets,
  Zap
} from 'lucide-react';
import { Button } from '../common/Button';
import { Badge, RiskBadge } from '../common/Badge';
import { PredictionBreakdown } from './PredictionBreakdown';
import { DiseaseInfo } from './DiseaseInfo';
import { RecommendationCard } from './RecommendationCard';
import { getDiseaseById, getDiseaseByName } from '../../data/diseases';

export const ResultDashboard = ({ image, result, onReset }) => {
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [activeBoxIndex, setActiveBoxIndex] = useState(null);

  if (!result) return null;

  const {
    disease,
    confidence = 94.6,
    status = "Disease Detected",
    risk = "Moderate",
    predictions = [],
    predicted_diseases = [],
    detections = [],
    is_healthy = false,
    is_multiple_diseases = false,
    summary = "",
    diseaseInfo = null,
    execution_time_ms = 128,
    executionTimeMs = 128,
    model_version = "MangoLeaf-DualVision-v3.0",
    modelVersion = "MangoLeaf-DualVision-v3.0"
  } = result;

  const execTime = execution_time_ms || executionTimeMs;
  const activeModel = model_version || modelVersion;
  const isHealthy = is_healthy || disease?.toLowerCase() === 'healthy';

  const getDiseaseColor = (diseaseId) => {
    switch (diseaseId?.toLowerCase()) {
      case 'anthracnose':
        return { border: 'border-amber-400', bg: 'bg-amber-500/20', text: 'text-amber-300', fill: '#f59e0b' };
      case 'bacterial-canker':
        return { border: 'border-rose-400', bg: 'bg-rose-500/20', text: 'text-rose-300', fill: '#f43f5e' };
      case 'powdery-mildew':
        return { border: 'border-purple-400', bg: 'bg-purple-500/20', text: 'text-purple-300', fill: '#c084fc' };
      case 'sooty-mold':
        return { border: 'border-slate-400', bg: 'bg-slate-500/20', text: 'text-slate-300', fill: '#94a3b8' };
      case 'die-back':
        return { border: 'border-orange-400', bg: 'bg-orange-500/20', text: 'text-orange-300', fill: '#fb923c' };
      case 'gall-midge':
        return { border: 'border-yellow-400', bg: 'bg-yellow-500/20', text: 'text-yellow-300', fill: '#facc15' };
      case 'cutting-weevil':
        return { border: 'border-pink-400', bg: 'bg-pink-500/20', text: 'text-pink-300', fill: '#f472b6' };
      default:
        return { border: 'border-emerald-400', bg: 'bg-emerald-500/20', text: 'text-emerald-300', fill: '#34d399' };
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Top Banner Navigation & Reset */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <button
            onClick={onReset}
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700 transition-colors"
            title="Return to upload"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <h2 className="text-xl sm:text-2xl font-black text-white tracking-tight flex items-center gap-2">
              <span>Detection Result</span>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            </h2>
            <p className="text-xs text-slate-400">
              Inference completed in <span className="text-emerald-400 font-mono">{execTime}ms</span> • Engine: <span className="font-mono text-slate-300">{activeModel}</span>
            </p>
          </div>
        </div>

        <Button
          variant="secondary"
          size="md"
          icon={RefreshCw}
          onClick={onReset}
          className="self-start sm:self-auto"
        >
          Analyze Another Leaf
        </Button>
      </div>

      {/* MULTIPLE DISEASES DETECTED CALLOUT BANNER */}
      {is_multiple_diseases && (
        <div className="rounded-3xl p-6 sm:p-7 border-2 border-amber-500/40 bg-gradient-to-r from-amber-950/50 via-slate-900/90 to-rose-950/40 shadow-2xl relative overflow-hidden space-y-4 animate-in fade-in">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-start sm:items-center gap-3.5">
              <div className="w-12 h-12 rounded-2xl bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-amber-400 flex-shrink-0 shadow-lg">
                <AlertTriangle className="w-6 h-6 animate-bounce" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-300 text-[10px] font-extrabold uppercase tracking-widest border border-amber-500/30">
                    Multiple Pathologies
                  </span>
                </div>
                <h3 className="text-xl sm:text-2xl font-black text-white tracking-tight mt-1">
                  Multiple Diseases Detected on Single Leaf
                </h3>
                <p className="text-xs sm:text-sm text-slate-300 mt-0.5">
                  Distinct pathological lesion regions have been localized across the leaf blade.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 self-start sm:self-center">
              {predicted_diseases.map((dis, idx) => {
                const colors = getDiseaseColor(dis.disease_id || dis.id);
                return (
                  <div
                    key={idx}
                    className={`px-3.5 py-2 rounded-xl bg-slate-900/90 border ${colors.border} flex items-center gap-2 shadow-md`}
                  >
                    <span className={`w-2 h-2 rounded-full ${colors.bg}`} />
                    <span className={`text-xs font-bold ${colors.text}`}>
                      {dis.name || dis.disease}
                    </span>
                    <span className="text-[11px] font-mono text-white font-black bg-slate-800 px-1.5 py-0.5 rounded">
                      {dis.confidence || dis.cnn_confidence}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Main Grid: Uploaded Leaf Specimen + Primary Prediction */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Specimen Preview with Bounding Box Overlay */}
        <div className="lg:col-span-5 space-y-6 lg:sticky lg:top-24">
          <div className="rounded-3xl glass-panel p-5 border border-slate-800 shadow-2xl space-y-4">
            <div className="flex items-center justify-between text-xs pb-2 border-b border-slate-800/80">
              <span className="font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Crosshair className="w-3.5 h-3.5 text-emerald-400" />
                Scanned Specimen & Detection Boxes
              </span>
              {detections.length > 0 && (
                <button
                  type="button"
                  onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                  className="flex items-center gap-1 text-[11px] text-emerald-400 hover:text-emerald-300 bg-slate-900 px-2 py-1 rounded-md border border-slate-800 hover:border-emerald-500/30 transition-colors"
                >
                  {showBoundingBoxes ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                  <span>{showBoundingBoxes ? 'Hide Boxes' : 'Show Boxes'}</span>
                </button>
              )}
            </div>

            {/* Specimen Viewport Container */}
            <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 aspect-4/3 flex items-center justify-center group shadow-inner select-none">
              {/* Main Leaf Image */}
              <img
                src={image?.previewUrl}
                alt={image?.name || "Analyzed Mango Leaf"}
                className="w-full h-full object-contain"
              />

              {/* BOUNDING BOX OVERLAY LAYER */}
              {showBoundingBoxes && detections && detections.length > 0 && (
                <div className="absolute inset-0 pointer-events-none">
                  {detections.map((det, idx) => {
                    const [rx1, ry1, rx2, ry2] = det.relative_bbox || [0.2, 0.2, 0.6, 0.6];
                    const left = `${rx1 * 100}%`;
                    const top = `${ry1 * 100}%`;
                    const width = `${Math.max(8, (rx2 - rx1) * 100)}%`;
                    const height = `${Math.max(8, (ry2 - ry1) * 100)}%`;
                    const colors = getDiseaseColor(det.disease_id);
                    const isActive = activeBoxIndex === idx;

                    return (
                      <div
                        key={idx}
                        className={`absolute border-2 ${colors.border} rounded-lg transition-all duration-300 ${
                          isActive ? 'ring-4 ring-emerald-400/40 scale-[1.02] z-20' : 'z-10'
                        } ${colors.bg}`}
                        style={{ left, top, width, height }}
                        onMouseEnter={() => setActiveBoxIndex(idx)}
                        onMouseLeave={() => setActiveBoxIndex(null)}
                      >
                        {/* Box Label Tag */}
                        <div
                          className={`absolute -top-6 left-0 px-2 py-0.5 rounded-md bg-slate-950/95 border ${colors.border} shadow-lg flex items-center gap-1.5 whitespace-nowrap backdrop-blur-md`}
                        >
                          <span className={`w-1.5 h-1.5 rounded-full ${colors.bg}`} />
                          <span className={`text-[10px] font-bold ${colors.text}`}>
                            {det.disease}
                          </span>
                          <span className="text-[9px] font-mono text-white font-extrabold">
                            {det.confidence}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Status Badge in Top Left */}
              <div className="absolute top-3 left-3 z-10 pointer-events-none">
                <RiskBadge risk={risk} />
              </div>

              {/* Bottom Tag */}
              <div className="absolute bottom-3 right-3 px-2.5 py-1 rounded-full bg-slate-950/80 border border-slate-700 text-[11px] font-mono text-slate-300 backdrop-blur-md z-10 pointer-events-none">
                {detections.length > 0 ? `${detections.length} Diseased Region(s)` : '0 Lesions'}
              </div>
            </div>

            {/* Bounding Boxes Summary List if available */}
            {detections.length > 0 && (
              <div className="space-y-2 pt-1">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Detected Lesion Regions ({detections.length})
                </span>
                <div className="space-y-1.5">
                  {detections.map((det, idx) => {
                    const colors = getDiseaseColor(det.disease_id);
                    return (
                      <div
                        key={idx}
                        onMouseEnter={() => setActiveBoxIndex(idx)}
                        onMouseLeave={() => setActiveBoxIndex(null)}
                        className={`flex items-center justify-between p-2 rounded-xl bg-slate-900/80 border text-xs cursor-pointer transition-colors ${
                          activeBoxIndex === idx ? 'border-emerald-400 bg-slate-800' : 'border-slate-800/80'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span className={`w-2.5 h-2.5 rounded-md border ${colors.border} ${colors.bg}`} />
                          <span className="font-semibold text-slate-200">
                            Region {idx + 1}: {det.disease}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-[11px] text-emerald-400 font-bold">
                            {det.confidence}%
                          </span>
                          <span className="text-[10px] font-mono text-slate-500">
                            [{det.bbox.join(', ')}]
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Specimen File Info */}
            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800/80 text-xs text-slate-400 space-y-1.5">
              <div className="flex items-center justify-between">
                <span>File Name:</span>
                <span className="text-slate-200 font-mono font-medium truncate max-w-[200px]" title={image?.name}>
                  {image?.name || 'leaf_scan.jpg'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>Classification Status:</span>
                <span className={`font-semibold ${isHealthy ? 'text-emerald-400' : (is_multiple_diseases ? 'text-amber-300 font-extrabold' : 'text-amber-400')}`}>
                  {status}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Results & Breakdown */}
        <div className="lg:col-span-7 space-y-6">
          {/* Primary Prediction Hero Banner */}
          <div className="rounded-3xl glass-panel p-6 sm:p-8 border border-emerald-500/30 bg-gradient-to-br from-emerald-950/30 via-slate-900/60 to-slate-950 shadow-2xl relative overflow-hidden">
            <div className="absolute -top-12 -right-12 w-48 h-48 bg-emerald-500/10 rounded-full blur-2xl pointer-events-none" />

            <div className="relative space-y-5">
              {/* Status Header */}
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
                  <Sparkles className="w-4 h-4" />
                  {is_multiple_diseases ? "Multiple Pathologies Detected" : "Diagnostic Result"}
                </span>
                <div className="flex items-center gap-2">
                  <Badge variant={isHealthy ? 'success' : 'warning'} size="sm">
                    {status}
                  </Badge>
                  <RiskBadge risk={risk} />
                </div>
              </div>

              {/* Disease Name & Confidence Score Display */}
              <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-3 pt-1">
                <div>
                  <h3 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
                    {is_multiple_diseases && predicted_diseases.length > 0
                      ? predicted_diseases.map(d => d.name || d.disease).join(" + ")
                      : disease}
                  </h3>
                  {diseaseInfo?.scientificName && !is_multiple_diseases && (
                    <p className="text-xs sm:text-sm text-slate-400 italic font-mono mt-1">
                      {diseaseInfo.scientificName}
                    </p>
                  )}
                  {is_multiple_diseases && (
                    <p className="text-xs sm:text-sm text-slate-400 font-mono mt-1">
                      {predicted_diseases.length} distinct pathologies localized on specimen
                    </p>
                  )}
                </div>

                <div className="text-left sm:text-right bg-slate-900/80 sm:bg-transparent p-3 sm:p-0 rounded-xl sm:rounded-none border border-slate-800 sm:border-0">
                  <span className="text-xs uppercase font-semibold text-slate-400 block sm:inline">
                    {is_multiple_diseases ? "Max Confidence" : "Estimated Confidence"}
                  </span>
                  <div className="text-3xl sm:text-4xl font-black text-emerald-400 font-mono">
                    {typeof confidence === 'number' ? confidence.toFixed(1) : confidence}%
                  </div>
                </div>
              </div>

              {/* Prediction Summary Callout */}
              <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-xs sm:text-sm text-slate-300 leading-relaxed">
                <span className="font-bold text-slate-200">Diagnostic Summary: </span>
                {summary || `The model predicts ${disease} with an estimated confidence of ${confidence}%.`}
              </div>
            </div>
          </div>

          {/* Multi-Class Prediction Breakdown Component */}
          <PredictionBreakdown
            predictions={predictions}
            predictedDiseases={predicted_diseases}
          />

          {/* Disease Information & Recommendations */}
          {is_multiple_diseases && predicted_diseases.length > 1 ? (
            predicted_diseases.map((d, i) => {
              const info = getDiseaseById(d.disease_id || d.id) || getDiseaseByName(d.name || d.disease) || d;
              return (
                <div key={d.disease_id || i} className="space-y-6">
                  <DiseaseInfo disease={info} />
                  <RecommendationCard disease={info} />
                </div>
              );
            })
          ) : (
            <>
              <DiseaseInfo disease={diseaseInfo} />
              <RecommendationCard disease={diseaseInfo} />
            </>
          )}

          {/* Bottom Action Footer */}
          <div className="pt-4 flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-xs text-slate-500 text-center sm:text-left">
              Test another leaf? Upload a new photo or select another sample specimen.
            </p>
            <Button
              variant="primary"
              size="lg"
              icon={RefreshCw}
              onClick={onReset}
              className="w-full sm:w-auto shadow-xl shadow-emerald-950/80"
            >
              Analyze Another Leaf
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

import React, { useState } from 'react';
import { Sparkles, Activity, ShieldCheck, Info } from 'lucide-react';
import { UploadBox } from '../components/detection/UploadBox';
import { ImagePreview } from '../components/detection/ImagePreview';
import { AnalysisLoader } from '../components/detection/AnalysisLoader';
import { ResultDashboard } from '../components/detection/ResultDashboard';
import { Toast } from '../components/common/Toast';
import { predictMangoLeafDisease } from '../services/predictionService';

export const Detection = ({ initialSampleId, onClearInitialSample }) => {
  // Stages: 'idle' | 'selected' | 'analyzing' | 'result'
  const [stage, setStage] = useState('idle');
  const [selectedImage, setSelectedImage] = useState(null);
  const [analysisStepIndex, setAnalysisStepIndex] = useState(0);
  const [predictionResult, setPredictionResult] = useState(null);
  const [toastMessage, setToastMessage] = useState(null);

  // Handle image selected from upload or sample picker
  const handleImageSelected = (imageData) => {
    setSelectedImage(imageData);
    setStage('selected');
    setPredictionResult(null);
  };

  // Trigger analysis simulation
  const handleStartAnalysis = async () => {
    if (!selectedImage) return;

    setStage('analyzing');
    setAnalysisStepIndex(0);

    try {
      const result = await predictMangoLeafDisease(
        selectedImage.isSample ? selectedImage : selectedImage.file || selectedImage,
        (stepIdx) => {
          setAnalysisStepIndex(stepIdx);
        }
      );

      setPredictionResult(result);
      setStage('result');
    } catch (err) {
      console.error('Analysis error:', err);
      setToastMessage('An error occurred during pattern classification. Please try again.');
      setStage('selected');
    }
  };

  // Reset entire detection flow
  const handleReset = () => {
    setSelectedImage(null);
    setPredictionResult(null);
    setStage('idle');
    setAnalysisStepIndex(0);
    if (onClearInitialSample) onClearInitialSample();
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-8">
      {/* Toast Notification */}
      <Toast
        message={toastMessage}
        type="error"
        onClose={() => setToastMessage(null)}
      />

      {/* Page Header (Only show on idle and selected stages) */}
      {stage !== 'result' && stage !== 'analyzing' && (
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/70 border border-emerald-500/30 text-emerald-300 text-xs font-semibold shadow-md">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            <span>AI Diagnostic Engine</span>
          </div>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-white tracking-tight">
            Analyze Your Mango Leaf
          </h1>
          <p className="text-slate-400 text-base sm:text-lg">
            Upload an image of a mango leaf to identify the most likely disease category.
          </p>
        </div>
      )}

      {/* Stage: Idle (Upload Area) */}
      {stage === 'idle' && (
        <div className="max-w-4xl mx-auto">
          <UploadBox
            onImageSelected={handleImageSelected}
            onError={(msg) => setToastMessage(msg)}
          />
        </div>
      )}

      {/* Stage: Selected (Image Preview & Metadata) */}
      {stage === 'selected' && (
        <div className="max-w-4xl mx-auto">
          <ImagePreview
            image={selectedImage}
            onRemove={handleReset}
            onAnalyze={handleStartAnalysis}
            isAnalyzing={false}
          />
        </div>
      )}

      {/* Stage: Analyzing (Laser Scan & Step Progress Animation) */}
      {stage === 'analyzing' && (
        <AnalysisLoader
          previewUrl={selectedImage?.previewUrl}
          activeStepIndex={analysisStepIndex}
        />
      )}

      {/* Stage: Result (Complete Result Dashboard) */}
      {stage === 'result' && (
        <ResultDashboard
          image={selectedImage}
          result={predictionResult}
          onReset={handleReset}
        />
      )}
    </div>
  );
};

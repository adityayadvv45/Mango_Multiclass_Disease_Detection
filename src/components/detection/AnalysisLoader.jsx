import React from 'react';
import { Loader2, Check, Sparkles, Cpu, Scan } from 'lucide-react';

export const AnalysisLoader = ({ previewUrl, activeStepIndex = 1 }) => {
  const steps = [
    { label: 'Image uploaded', description: 'Validated dimensions & color space' },
    { label: 'Image prepared', description: 'Normalized tensor matrix to 224×224' },
    { label: 'Detecting disease pattern', description: 'Extracting convolutional feature maps' },
    { label: 'Generating prediction', description: 'Computing multi-class softmax distribution' }
  ];

  return (
    <div className="rounded-3xl glass-panel p-8 sm:p-12 border border-slate-700/80 shadow-2xl max-w-3xl mx-auto space-y-8 animate-in fade-in duration-300">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-emerald-950/80 border border-emerald-500/30 text-emerald-300 text-xs font-semibold">
          <Loader2 className="w-3.5 h-3.5 animate-spin text-emerald-400" />
          <span>ML Inference in Progress</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Analyzing Leaf...
        </h2>
        <p className="text-sm text-slate-400">
          Extracting morphological features and cross-referencing disease patterns
        </p>
      </div>

      {/* Leaf Scanner Visual with Active Laser */}
      <div className="relative max-w-md mx-auto aspect-video rounded-2xl overflow-hidden bg-slate-950 border-2 border-emerald-500/40 shadow-2xl flex items-center justify-center">
        {previewUrl && (
          <img
            src={previewUrl}
            alt="Leaf under active neural scan"
            className="w-full h-full object-contain filter brightness-90"
          />
        )}

        {/* Laser Line */}
        <div className="laser-line animate-scan" />

        {/* Corner HUD Markers */}
        <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-emerald-400" />
        <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-emerald-400" />
        <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-emerald-400" />
        <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-emerald-400" />

        {/* Central Scan Badge */}
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-slate-950/90 border border-emerald-500/40 text-[11px] font-mono text-emerald-300 flex items-center gap-2 backdrop-blur-md">
          <Scan className="w-3.5 h-3.5 animate-pulse" />
          <span>Scanning Lesions & Textures...</span>
        </div>
      </div>

      {/* Step-by-Step Progress Checklist */}
      <div className="bg-slate-900/80 rounded-2xl p-6 border border-slate-800 space-y-4 max-w-lg mx-auto">
        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 pb-2 border-b border-slate-800">
          Inference Pipeline Execution
        </h4>
        <div className="space-y-3.5">
          {steps.map((step, idx) => {
            const isCompleted = idx < activeStepIndex;
            const isCurrent = idx === activeStepIndex;
            const isPending = idx > activeStepIndex;

            return (
              <div key={idx} className="flex items-start gap-3.5">
                {/* Status Indicator Icon */}
                <div className="mt-0.5 flex-shrink-0">
                  {isCompleted && (
                    <div className="w-5 h-5 rounded-full bg-emerald-500 text-slate-950 flex items-center justify-center font-bold shadow-md shadow-emerald-950">
                      <Check className="w-3.5 h-3.5 stroke-[3]" />
                    </div>
                  )}
                  {isCurrent && (
                    <div className="w-5 h-5 rounded-full bg-emerald-950 border border-emerald-400 text-emerald-300 flex items-center justify-center">
                      <div className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                    </div>
                  )}
                  {isPending && (
                    <div className="w-5 h-5 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-600">
                      <div className="w-1.5 h-1.5 rounded-full bg-slate-600" />
                    </div>
                  )}
                </div>

                {/* Step Text */}
                <div className="flex-1">
                  <div className={`text-sm font-semibold flex items-center gap-2 ${
                    isCompleted ? 'text-slate-200' : isCurrent ? 'text-emerald-300' : 'text-slate-500'
                  }`}>
                    <span>{step.label}</span>
                    {isCurrent && (
                      <span className="text-[10px] uppercase font-mono px-1.5 py-0.2 bg-emerald-500/20 text-emerald-300 rounded">
                        Active
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-500">{step.description}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

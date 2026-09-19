import React, { useState, useEffect } from 'react';
import { CheckCircle2, Circle, Loader2, Sparkles, Cpu, Scan } from 'lucide-react';

export default function AnalysisLoader({ previewUrl, onComplete }) {
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    { label: "Image uploaded", sub: "Validated dimensions & color space" },
    { label: "Image prepared", sub: "Normalized to 224×224 RGB tensor" },
    { label: "Neural feature extraction", sub: "Multi-scale convolutional filtering" },
    { label: "Softmax probability distribution", sub: "Localizing lesion bounding boxes" },
  ];

  useEffect(() => {
    // Progressive step progression over ~2.4 seconds
    const timer1 = setTimeout(() => setCurrentStep(1), 500);
    const timer2 = setTimeout(() => setCurrentStep(2), 1100);
    const timer3 = setTimeout(() => setCurrentStep(3), 1700);
    const timer4 = setTimeout(() => {
      if (onComplete) onComplete();
    }, 2400);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
    };
  }, [onComplete]);

  return (
    <div className="w-full max-w-2xl mx-auto rounded-3xl bg-slate-900/80 border border-slate-800 p-8 md:p-12 shadow-2xl space-y-8 animate-fadeIn text-center">
      
      {/* Top Header */}
      <div className="space-y-2">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
          <span>ML Inference in Progress</span>
        </div>
        <h3 className="text-3xl font-black text-white font-heading">
          Analyzing Leaf...
        </h3>
        <p className="text-sm text-slate-400">
          Extracting morphological features and cross-referencing disease patterns
        </p>
      </div>

      {/* Center Laser Scanning Specimen Viewport */}
      <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-emerald-500/30 aspect-video max-h-[300px] mx-auto w-full flex items-center justify-center shadow-inner">
        {previewUrl && (
          <img
            src={previewUrl}
            alt="Scanning Leaf"
            className="w-full h-full object-contain p-4 opacity-85 filter contrast-125 select-none"
          />
        )}

        {/* Animated Laser Scanline */}
        <div className="absolute inset-x-0 h-1.5 bg-gradient-to-r from-transparent via-emerald-400 to-transparent shadow-[0_0_25px_#10b981] animate-scanline z-20"></div>

        {/* Reticle Focus Lines */}
        <div className="absolute inset-4 border border-emerald-500/20 pointer-events-none"></div>

        {/* Scanning Badge */}
        <div className="absolute bottom-3 bg-slate-950/90 backdrop-blur-md px-4 py-1.5 rounded-full border border-emerald-500/40 text-[11px] font-mono text-emerald-400 font-semibold tracking-wider flex items-center gap-2 shadow-lg">
          <Scan className="w-3.5 h-3.5 animate-pulse" />
          <span>Scanning Lesions & Textures...</span>
        </div>
      </div>

      {/* Inference Pipeline Checklist */}
      <div className="space-y-4 text-left max-w-lg mx-auto pt-2">
        <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
          INFERENCE PIPELINE EXECUTION
        </div>

        <div className="space-y-2.5">
          {steps.map((step, idx) => {
            const isCompleted = idx < currentStep;
            const isCurrent = idx === currentStep;

            return (
              <div
                key={idx}
                className={`flex items-center justify-between p-3 rounded-xl border transition-all duration-300 ${
                  isCompleted
                    ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300'
                    : isCurrent
                    ? 'bg-slate-800/90 border-emerald-500/50 text-white shadow-md'
                    : 'bg-slate-950/30 border-slate-800/40 text-slate-400 opacity-60'
                }`}
              >
                <div className="flex items-center gap-3">
                  {isCompleted ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                  ) : isCurrent ? (
                    <div className="w-4 h-4 flex items-center justify-center flex-shrink-0">
                      <Loader2 className="w-4 h-4 text-emerald-400 animate-spin" />
                    </div>
                  ) : (
                    <Circle className="w-4 h-4 text-slate-400 flex-shrink-0" />
                  )}
                  <div>
                    <div className="text-xs font-semibold leading-tight">
                      {step.label}
                    </div>
                    <div className="text-[10px] text-slate-400">
                      {step.sub}
                    </div>
                  </div>
                </div>

                {isCompleted && (
                  <span className="text-[10px] font-mono font-bold text-emerald-400">
                    DONE
                  </span>
                )}
                {isCurrent && (
                  <span className="text-[10px] font-mono font-bold text-amber-400 animate-pulse">
                    RUNNING
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
}

import React from 'react';
import { Lightbulb, RotateCcw, ShieldAlert, AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function RecommendationCard({ disease, onReset }) {
  if (!disease) return null;

  const steps = disease.recommendedSteps || [
    "Prune and destroy infected foliage to prevent spore buildup in orchard soil.",
    "Apply appropriate certified fungicides or biological controls at first flush.",
    "Improve orchard drainage and canopy airflow through selective pruning.",
    "Consult local agricultural extension specialists before applying chemical sprays."
  ];

  return (
    <div className="w-full rounded-2xl bg-slate-900/60 border border-slate-800 p-5 md:p-6 space-y-5 shadow-xl">
      
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <Lightbulb className="w-4 h-4 text-emerald-400" />
          <h3 className="text-base md:text-lg font-bold text-white font-heading">
            Recommended Next Steps
          </h3>
        </div>
        <span className="text-[11px] text-slate-400">
          Suggested cultural & orchard management guidance
        </span>
      </div>

      {/* 4-Step Numbered Guidance List */}
      <div className="space-y-2.5">
        {steps.map((step, index) => (
          <div
            key={index}
            className="flex items-start gap-3.5 p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 text-xs md:text-sm text-slate-200"
          >
            <div className="w-6 h-6 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center justify-center font-bold text-xs flex-shrink-0 mt-0.5">
              {index + 1}
            </div>
            <p className="leading-relaxed">
              {step}
            </p>
          </div>
        ))}
      </div>

      {/* Important Advisory Box */}
      <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200/90 flex items-start gap-3">
        <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-bold text-amber-300 block">Important Advisory:</span>
          <p className="text-slate-300 leading-normal">
            Monitor the affected leaf and consider appropriate plant disease management practices. For reliable field diagnosis, confirm the result with an agricultural expert or university extension center before applying chemical treatments.
          </p>
        </div>
      </div>

      {/* Reset CTA Footer */}
      {onReset && (
        <div className="pt-3 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-xs text-slate-400 text-center sm:text-left">
            Test another leaf? Upload a new photo or select another sample specimen.
          </div>

          <button
            onClick={onReset}
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs shadow-lg shadow-emerald-500/20 hover:scale-[1.02] active:scale-100 transition-all duration-200"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Analyze Another Leaf</span>
          </button>
        </div>
      )}

    </div>
  );
}

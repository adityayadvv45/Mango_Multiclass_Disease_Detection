import React from 'react';
import { ShieldAlert, CheckCircle, ArrowRight, Stethoscope, AlertCircle } from 'lucide-react';

export const RecommendationCard = ({ disease }) => {
  if (!disease) return null;

  const actions = disease.recommendedActions || [
    "Monitor the affected foliage and isolate severe infections.",
    "Improve orchard sanitation and air ventilation through selective canopy pruning.",
    "Consult certified agricultural extension officers for verified diagnostic confirmation."
  ];

  return (
    <div className="rounded-2xl glass-panel p-6 border border-slate-800 space-y-5">
      {/* Header */}
      <div className="flex items-center gap-2.5 pb-3 border-b border-slate-800">
        <div className="w-8 h-8 rounded-lg bg-emerald-950/80 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
          <Stethoscope className="w-4 h-4" />
        </div>
        <div>
          <h4 className="text-sm font-bold text-white uppercase tracking-wider">
            Recommended Next Steps
          </h4>
          <p className="text-[11px] text-slate-400">
            Suggested cultural & orchard management guidance
          </p>
        </div>
      </div>

      {/* Structured Recommendation Action List */}
      <div className="space-y-2.5">
        {actions.map((action, idx) => (
          <div
            key={idx}
            className="flex items-start gap-3 p-3.5 rounded-xl bg-slate-900/90 border border-slate-800/90 text-xs sm:text-sm text-slate-200"
          >
            <div className="w-5 h-5 rounded-full bg-emerald-500/10 text-emerald-400 flex items-center justify-center font-bold text-xs flex-shrink-0 mt-0.5 border border-emerald-500/20">
              {idx + 1}
            </div>
            <p className="leading-relaxed flex-1">{action}</p>
          </div>
        ))}
      </div>

      {/* Advisory Disclaimer Notice */}
      <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/20 text-xs text-amber-200/90 flex items-start gap-3">
        <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
        <p className="leading-relaxed text-[11px] sm:text-xs">
          <strong>Important Advisory:</strong> Monitor the affected leaf and consider appropriate plant disease management practices. For reliable field diagnosis, confirm the result with an agricultural expert or university extension center before applying chemical treatments.
        </p>
      </div>
    </div>
  );
};

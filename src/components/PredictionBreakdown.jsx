import React from 'react';
import { BarChart3, CheckCircle2 } from 'lucide-react';
import { DISEASE_BOX_COLORS } from '../data/mockResults';

export default function PredictionBreakdown({ predictions = [], primaryDiseaseName = "" }) {
  if (!predictions || predictions.length === 0) return null;

  return (
    <div className="w-full rounded-2xl bg-slate-900/60 border border-slate-800 p-5 md:p-6 space-y-4 shadow-xl">
      
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-emerald-400" />
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Multi-Class Prediction Breakdown
          </h4>
        </div>
        <span className="text-[11px] text-slate-400 font-mono">
          Model Detection Scores
        </span>
      </div>

      <p className="text-xs text-slate-400">
        Calibrated probability across all 8 supported botanical categories:
      </p>

      {/* Progress Bars Grid */}
      <div className="space-y-3 pt-1">
        {predictions.map((item, idx) => {
          const colorInfo = DISEASE_BOX_COLORS[item.name] || { stroke: "#10b981" };
          const isDetected = item.isDetected || item.confidence > 75;

          return (
            <div key={idx} className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span 
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: colorInfo.stroke }}
                  ></span>
                  <span className={`font-semibold ${isDetected ? 'text-white' : 'text-slate-400'}`}>
                    {item.name}
                  </span>
                  {isDetected && (
                    <span className="px-2 py-0.2 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[10px] font-bold">
                      Detected
                    </span>
                  )}
                </div>

                <span className="font-mono font-bold text-slate-300">
                  {item.confidence.toFixed(1)}%
                </span>
              </div>

              {/* Progress Track */}
              <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
                <div
                  className="h-full rounded-full transition-all duration-1000 ease-out"
                  style={{
                    width: `${Math.min(100, Math.max(2, item.confidence))}%`,
                    backgroundColor: isDetected ? colorInfo.stroke : '#475569'
                  }}
                ></div>
              </div>
            </div>
          );
        })}
      </div>

    </div>
  );
}

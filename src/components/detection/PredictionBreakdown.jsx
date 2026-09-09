import React from 'react';
import { BarChart3, TrendingUp, Layers } from 'lucide-react';

export const PredictionBreakdown = ({ predictions = [], predictedDiseases = [] }) => {
  if (!predictions || predictions.length === 0) return null;

  const detectedNames = new Set((predictedDiseases || []).map((d) => d.name || d.disease));
  const isMulti = detectedNames.size > 1;

  return (
    <div className="rounded-2xl glass-panel p-6 border border-slate-800 space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-emerald-400" />
          <h4 className="text-sm font-bold text-white uppercase tracking-wider">
            Multi-Class Prediction Breakdown
          </h4>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          Model Detection Scores
        </span>
      </div>

      <p className="text-xs text-slate-400">
        Calibrated probability across all {predictions.length} supported botanical categories:
      </p>

      {/* Progress Bars List */}
      <div className="space-y-3.5 pt-1">
        {predictions.map((item, index) => {
          const isDetected = detectedNames.has(item.name);
          const isTop = index === 0 && isDetected;
          const isAlsoDetected = !isTop && isDetected;
          const confidence = parseFloat(item.confidence) || 0;

          return (
            <div key={item.name || index} className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${
                    isTop
                      ? 'bg-emerald-400 ring-4 ring-emerald-500/20'
                      : (isAlsoDetected ? 'bg-amber-400 ring-4 ring-amber-500/20' : 'bg-slate-600')
                  }`} />
                  <span className={`font-medium ${
                    isTop ? 'text-white font-bold' : (isAlsoDetected ? 'text-amber-300 font-semibold' : 'text-slate-300')
                  }`}>
                    {item.name}
                  </span>
                  {isTop && (
                    <span className="px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[10px] font-semibold">
                      {isMulti ? "Detected" : "Primary Match"}
                    </span>
                  )}
                  {isAlsoDetected && (
                    <span className="px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 text-[10px] font-semibold">
                      Also Detected
                    </span>
                  )}
                </div>
                <span className={`font-mono text-xs ${
                  isTop ? 'text-emerald-400 font-extrabold' : (isAlsoDetected ? 'text-amber-400 font-bold' : 'text-slate-400')
                }`}>
                  {confidence.toFixed(1)}%
                </span>
              </div>

              {/* Progress Bar Track */}
              <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden p-0.5 border border-slate-800">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ease-out ${
                    isTop
                      ? 'bg-gradient-to-r from-emerald-500 to-teal-400 shadow-md shadow-emerald-500/50'
                      : (isAlsoDetected
                          ? 'bg-gradient-to-r from-amber-500 to-orange-400'
                          : 'bg-slate-600/80')
                  }`}
                  style={{ width: `${Math.max(confidence, 1.5)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

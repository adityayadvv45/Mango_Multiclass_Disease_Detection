import React, { useState } from 'react';
import { BarChart3, PieChart as PieChartIcon, Sparkles } from 'lucide-react';
import { DISEASE_BOX_COLORS } from '../data/mockResults';

export default function PredictionBreakdown({ predictions = [] }) {
  const [viewMode, setViewMode] = useState('pie'); // 'pie' | 'bars'
  const [hoveredIdx, setHoveredIdx] = useState(null);

  if (!predictions || predictions.length === 0) return null;

  // Calculate SVG Pie/Donut Chart parameters
  const totalConfidence = predictions.reduce((sum, p) => sum + (p.confidence || 0), 0) || 100;
  const radius = 68;
  const cx = 80;
  const cy = 80;
  const strokeWidth = 24;
  const circumference = 2 * Math.PI * radius;

  const { slices: pieSlices } = predictions.reduce(
    (acc, item, idx) => {
      const val = item.confidence || 0;
      const fraction = val / totalConfidence;
      const strokeDasharray = `${fraction * circumference} ${circumference}`;
      const strokeDashoffset = -acc.cumulativeAngle * circumference;
      const colorInfo = DISEASE_BOX_COLORS[item.name] || { stroke: "#10b981" };
      const isDetected = item.isDetected || item.confidence > 75;

      acc.slices.push({
        ...item,
        color: colorInfo.stroke,
        strokeDasharray,
        strokeDashoffset,
        isDetected,
        idx
      });
      acc.cumulativeAngle += fraction;
      return acc;
    },
    { cumulativeAngle: 0, slices: [] }
  );

  const activeHoverItem = hoveredIdx !== null ? predictions[hoveredIdx] : null;

  return (
    <div className="w-full rounded-2xl bg-slate-900/70 border border-slate-800 p-5 md:p-6 space-y-4 shadow-xl backdrop-blur-sm">
      
      {/* Header & View Switcher */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          {viewMode === 'pie' ? (
            <PieChartIcon className="w-4 h-4 text-emerald-400" />
          ) : (
            <BarChart3 className="w-4 h-4 text-emerald-400" />
          )}
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Multi-Class Prediction Breakdown
          </h4>
        </div>
        
        {/* Toggle View Buttons */}
        <div className="flex items-center bg-slate-950 p-0.5 rounded-lg border border-slate-800">
          <button
            type="button"
            onClick={() => setViewMode('pie')}
            className={`flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium rounded-md transition-all ${
              viewMode === 'pie'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <PieChartIcon className="w-3.5 h-3.5" />
            <span>Pie Chart</span>
          </button>
          <button
            type="button"
            onClick={() => setViewMode('bars')}
            className={`flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium rounded-md transition-all ${
              viewMode === 'bars'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Bars</span>
          </button>
        </div>
      </div>

      <p className="text-xs text-slate-400">
        Calibrated probability across all 8 botanical categories:
      </p>

      {/* PIE CHART VIEW */}
      {viewMode === 'pie' && (
        <div className="pt-2 flex flex-col md:flex-row items-center gap-6">
          {/* SVG Donut Chart */}
          <div className="relative flex-shrink-0 flex items-center justify-center">
            <svg width="160" height="160" viewBox="0 0 160 160" className="transform -rotate-90">
              {/* Background Track */}
              <circle
                cx={cx}
                cy={cy}
                r={radius}
                fill="transparent"
                stroke="#1e293b"
                strokeWidth={strokeWidth}
              />
              
              {/* Pie Slices */}
              {pieSlices.map((slice) => {
                const isHovered = hoveredIdx === slice.idx;
                return (
                  <circle
                    key={slice.idx}
                    cx={cx}
                    cy={cy}
                    r={radius}
                    fill="transparent"
                    stroke={slice.color}
                    strokeWidth={isHovered ? strokeWidth + 4 : strokeWidth}
                    strokeDasharray={slice.strokeDasharray}
                    strokeDashoffset={slice.strokeDashoffset}
                    className="transition-all duration-300 cursor-pointer hover:opacity-100 opacity-90"
                    onMouseEnter={() => setHoveredIdx(slice.idx)}
                    onMouseLeave={() => setHoveredIdx(null)}
                  />
                );
              })}
            </svg>

            {/* Center Label */}
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none p-2">
              {activeHoverItem ? (
                <>
                  <span className="text-[10px] uppercase font-bold text-slate-400 truncate max-w-[85px]">
                    {activeHoverItem.name}
                  </span>
                  <span className="text-sm font-extrabold text-white font-mono">
                    {activeHoverItem.confidence.toFixed(1)}%
                  </span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 text-emerald-400 mb-0.5" />
                  <span className="text-[10px] font-bold text-slate-300">
                    8 Classes
                  </span>
                  <span className="text-[9px] text-slate-500 font-mono">
                    Softmax
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Legend Grid with all 8 Class Names */}
          <div className="w-full grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
            {predictions.map((item, idx) => {
              const colorInfo = DISEASE_BOX_COLORS[item.name] || { stroke: "#10b981" };
              const isDetected = item.isDetected || item.confidence > 75;
              const isHovered = hoveredIdx === idx;

              return (
                <div
                  key={idx}
                  onMouseEnter={() => setHoveredIdx(idx)}
                  onMouseLeave={() => setHoveredIdx(null)}
                  className={`flex items-center justify-between p-2 rounded-lg border transition-all cursor-pointer ${
                    isHovered
                      ? 'bg-slate-800/90 border-emerald-500/50 shadow-md'
                      : 'bg-slate-950/40 border-slate-800/60 hover:bg-slate-800/40'
                  }`}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: colorInfo.stroke }}
                    ></span>
                    <span className={`text-[11px] truncate ${isDetected ? 'font-bold text-white' : 'text-slate-300'}`}>
                      {item.name}
                    </span>
                  </div>

                  <div className="flex items-center gap-1.5 flex-shrink-0 pl-1">
                    {isDetected && (
                      <span className="px-1.5 py-0.2 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[9px] font-bold">
                        Top
                      </span>
                    )}
                    <span className="font-mono text-[11px] font-bold text-slate-200">
                      {item.confidence.toFixed(1)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* HORIZONTAL BARS VIEW */}
      {viewMode === 'bars' && (
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
      )}

    </div>
  );
}

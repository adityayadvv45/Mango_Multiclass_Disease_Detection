import React, { useState } from 'react';
import { Eye, EyeOff, Scan, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { DISEASE_BOX_COLORS } from '../data/mockResults';

export default function SpecimenBoxOverlay({ 
  imagePreviewUrl, 
  regions = [], 
  hoveredRegionId = null,
  onHoverRegion = () => {},
  isHealthy = false,
  totalLesionsCount = 0
}) {
  const [showBoxes, setShowBoxes] = useState(true);

  return (
    <div className="space-y-3">
      
      {/* Specimen Header & Hide/Show Switch */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2 font-bold text-slate-200 uppercase tracking-wider">
          <Scan className="w-4 h-4 text-emerald-400" />
          <span>Scanned Specimen & Detection Boxes</span>
        </div>

        {regions.length > 0 && (
          <button
            onClick={() => setShowBoxes(!showBoxes)}
            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-950 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-emerald-400 font-semibold transition-colors"
          >
            {showBoxes ? (
              <>
                <EyeOff className="w-3.5 h-3.5 text-slate-400" />
                <span>Hide Boxes</span>
              </>
            ) : (
              <>
                <Eye className="w-3.5 h-3.5 text-emerald-400" />
                <span>Show Boxes</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* Interactive Image Frame with Overlays */}
      <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 w-full min-h-[380px] max-h-[500px] flex items-center justify-center group shadow-2xl p-3">
        
        {/* Inner container tightly matching the image boundaries */}
        <div className="relative inline-block max-w-full max-h-full">
          {/* Base Specimen Image */}
          <img
            src={imagePreviewUrl}
            alt="Scanned Mango Leaf Specimen"
            className="max-w-full max-h-[460px] w-auto h-auto object-contain block rounded-lg select-none"
          />

          {/* Bounding Boxes Layer - aligned exactly to the image */}
          {showBoxes && regions.length > 0 && (
            <div className="absolute inset-0 pointer-events-none">
              {regions.map((region) => {
                const colorInfo = DISEASE_BOX_COLORS[region.disease] || DISEASE_BOX_COLORS["Anthracnose"];
                const isHovered = hoveredRegionId === region.id;
                const { top, left, width, height } = region.normBox || { top: 20, left: 20, width: 25, height: 25 };

                return (
                  <div
                    key={region.id}
                    onMouseEnter={() => onHoverRegion(region.id)}
                    onMouseLeave={() => onHoverRegion(null)}
                    style={{
                      top: `${top}%`,
                      left: `${left}%`,
                      width: `${width}%`,
                      height: `${height}%`,
                      borderColor: colorInfo.stroke,
                      backgroundColor: isHovered ? colorInfo.fill.replace('0.15', '0.35') : colorInfo.fill,
                    }}
                    className={`absolute rounded-md border-2 pointer-events-auto cursor-pointer transition-all duration-200 ${
                      isHovered ? 'scale-105 z-30 shadow-lg ring-2 ring-white/50' : 'z-10 hover:z-20'
                    }`}
                  >
                    {/* Bounding Box Label Tag */}
                    <div
                      style={{ backgroundColor: colorInfo.stroke, color: '#09090b' }}
                      className="absolute -top-5 left-0 px-1.5 py-0.5 rounded text-[9px] font-mono font-black whitespace-nowrap shadow-md uppercase tracking-tight flex items-center gap-1"
                    >
                      <span>{region.disease}</span>
                      <span className="opacity-90">{region.confidence}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Reticle Focus Corners */}
        <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-emerald-400/70 pointer-events-none"></div>
        <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-emerald-400/70 pointer-events-none"></div>
        <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-emerald-400/70 pointer-events-none"></div>
        <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-emerald-400/70 pointer-events-none"></div>

        {/* High Risk / Healthy Indicator Floating Badge */}
        <div className="absolute top-3 left-3 pointer-events-none">
          {isHealthy ? (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-950/80 backdrop-blur-md border border-emerald-500/40 text-emerald-300 text-[11px] font-bold">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>Optimal Leaf Blade</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-rose-950/80 backdrop-blur-md border border-rose-500/40 text-rose-300 text-[11px] font-bold animate-pulse">
              <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
              <span>High Risk</span>
            </div>
          )}
        </div>

        {/* Bottom Floating Diseased Regions Count Badge */}
        {!isHealthy && (
          <div className="absolute bottom-3 right-3 bg-slate-950/90 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-800 text-[11px] font-bold text-amber-300 shadow-xl pointer-events-none">
            {totalLesionsCount || regions.length} Diseased Region(s)
          </div>
        )}

      </div>

    </div>
  );
}

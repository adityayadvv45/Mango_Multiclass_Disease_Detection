import React from 'react';
import { Layers } from 'lucide-react';
import { DISEASE_BOX_COLORS } from '../data/mockResults';

export default function LesionRegionList({ 
  regions = [], 
  hoveredRegionId = null, 
  onHoverRegion = () => {},
  totalCount = 0
}) {
  if (!regions || regions.length === 0) return null;

  return (
    <div className="rounded-2xl bg-slate-950/70 border border-slate-800 p-4 space-y-3 shadow-inner">
      
      {/* Title */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
        <div className="flex items-center gap-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
          <Layers className="w-3.5 h-3.5 text-emerald-400" />
          <span>Detected Lesion Regions ({totalCount || regions.length})</span>
        </div>
        <span className="text-[10px] text-slate-400 font-mono">
          [ymin, xmin, ymax, xmax]
        </span>
      </div>

      {/* Scrollable list of regions */}
      <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1 custom-scrollbar">
        {regions.map((region) => {
          const isHovered = hoveredRegionId === region.id;
          const colorInfo = DISEASE_BOX_COLORS[region.disease] || DISEASE_BOX_COLORS["Anthracnose"];
          const coords = region.box ? `[${region.box.join(', ')}]` : `[${region.normBox?.top * 5}, ${region.normBox?.left * 5}, ...]`;

          return (
            <div
              key={region.id}
              onMouseEnter={() => onHoverRegion(region.id)}
              onMouseLeave={() => onHoverRegion(null)}
              className={`flex items-center justify-between p-2 rounded-lg text-xs font-mono transition-all duration-150 cursor-pointer ${
                isHovered
                  ? 'bg-slate-800 border border-slate-700 text-white shadow-sm'
                  : 'bg-slate-900/50 hover:bg-slate-850 border border-slate-850 text-slate-400'
              }`}
            >
              <div className="flex items-center gap-2 truncate">
                <span 
                  className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ backgroundColor: colorInfo.stroke }}
                ></span>
                <span className="text-slate-300 font-semibold truncate">
                  Region {region.id}: {region.disease}
                </span>
              </div>

              <div className="flex items-center gap-3 flex-shrink-0">
                <span className="text-emerald-400 font-bold">
                  {region.confidence}%
                </span>
                <span className="text-[10px] text-slate-400 hidden sm:inline">
                  {coords}
                </span>
              </div>
            </div>
          );
        })}
      </div>

    </div>
  );
}

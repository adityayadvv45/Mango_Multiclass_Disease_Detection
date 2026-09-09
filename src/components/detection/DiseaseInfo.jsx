import React from 'react';
import { BookOpen, CheckCircle2, AlertCircle, Info, Sparkles } from 'lucide-react';
import { Badge, RiskBadge } from '../common/Badge';

export const DiseaseInfo = ({ disease }) => {
  if (!disease) return null;

  return (
    <div className="rounded-2xl glass-panel p-6 border border-slate-800 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <BookOpen className="w-5 h-5 text-emerald-400" />
          <div>
            <h4 className="text-base font-bold text-white">
              About {disease.name}
            </h4>
            <p className="text-xs text-slate-400 italic font-mono">
              {disease.scientificName}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={disease.badgeVariant || 'default'} size="sm">
            {disease.category}
          </Badge>
          <RiskBadge risk={disease.riskLevel || disease.severity || 'Low'} />
        </div>
      </div>

      {/* Description */}
      <div className="space-y-2">
        <p className="text-sm text-slate-300 leading-relaxed">
          {disease.fullDescription || disease.shortDescription}
        </p>
      </div>

      {/* Common Visual Indicators */}
      {disease.visualIndicators && disease.visualIndicators.length > 0 && (
        <div className="space-y-3 pt-2">
          <h5 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            Common Visual Indicators
          </h5>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {disease.visualIndicators.map((indicator, idx) => (
              <div
                key={idx}
                className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-slate-300 leading-snug"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1 flex-shrink-0" />
                <span>{indicator}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Disclaimer note */}
      <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-900/50 border border-slate-800 text-[11px] text-slate-500">
        <Info className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
        <span>
          Information provided is for educational and advisory reference based on standard plant pathology guidelines.
        </span>
      </div>
    </div>
  );
};

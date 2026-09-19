import React from 'react';
import { BookOpen, CheckCircle2, AlertCircle, Sparkles, HelpCircle } from 'lucide-react';

export default function DiseaseInfo({ disease }) {
  if (!disease) return null;

  return (
    <div className="w-full rounded-2xl bg-slate-900/60 border border-slate-800 p-5 md:p-6 space-y-5 shadow-xl">
      
      {/* Header with Title & Metadata Badges */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-emerald-400" />
          <h3 className="text-base md:text-lg font-bold text-white font-heading">
            About {disease.name}
          </h3>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-slate-800 border border-slate-700 text-slate-300">
            {disease.category}
          </span>
          <span className={`px-2.5 py-1 rounded-md text-[11px] font-bold border ${disease.badgeBg || 'bg-slate-800 text-slate-300'}`}>
            • {disease.risk} Risk
          </span>
        </div>
      </div>

      {/* Scientific Name Subheading */}
      {disease.scientificName && (
        <div className="text-xs font-mono italic text-slate-400">
          {disease.scientificName}
        </div>
      )}

      {/* Disease Overview & Causes */}
      <div className="space-y-3 text-xs md:text-sm text-slate-300 leading-relaxed">
        <p>
          {disease.description || disease.shortDescription}
        </p>

        {disease.causes && (
          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-1">
            <div className="text-[11px] font-bold uppercase tracking-wider text-amber-400 flex items-center gap-1.5">
              <HelpCircle className="w-3.5 h-3.5 text-amber-400" />
              <span>Cause of Disease & Pathogen Etiology:</span>
            </div>
            <p className="text-xs text-slate-300 leading-normal">
              {disease.causes}
            </p>
          </div>
        )}
      </div>

      {/* Common Visual Indicators */}
      {disease.visualIndicators && disease.visualIndicators.length > 0 && (
        <div className="space-y-3 pt-2">
          <div className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-emerald-400">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Common Visual Indicators</span>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {disease.visualIndicators.map((indicator, index) => (
              <div 
                key={index}
                className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs text-slate-300 leading-snug"
              >
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
                <span>{indicator}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Educational Note */}
      <div className="pt-2 text-[11px] text-slate-400 flex items-center gap-2">
        <AlertCircle className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
        <span>Information provided is for educational and advisory reference based on standard plant pathology guidelines.</span>
      </div>

    </div>
  );
}

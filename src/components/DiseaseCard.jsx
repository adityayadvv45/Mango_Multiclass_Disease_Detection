import React from 'react';
import { 
  ShieldCheck, 
  AlertTriangle, 
  Biohazard, 
  Sparkles, 
  Layers, 
  Flame, 
  Bug,
  Scissors,
  CheckCircle, 
  ArrowRight,
  Info
} from 'lucide-react';

const iconMap = {
  ShieldCheck: ShieldCheck,
  AlertTriangle: AlertTriangle,
  Biohazard: Biohazard,
  Sparkles: Sparkles,
  Layers: Layers,
  Flame: Flame,
  Bug: Bug,
  Scissors: Scissors
};

export default function DiseaseCard({ disease, onSelectForTest, compact = false }) {
  const IconComponent = iconMap[disease.iconName] || Info;

  const categoryColorMap = {
    'Healthy': 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    'Fungal': 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    'Fungal (Secondary)': 'bg-slate-400/10 text-slate-300 border-slate-500/30',
    'Physiological / Fungal': 'bg-orange-500/10 text-orange-400 border-orange-500/30',
    'Bacterial': 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    'Pest / Entomological': 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
  };

  const riskBadgeColorMap = {
    'None': 'text-emerald-400 bg-emerald-950/60 border-emerald-800/60',
    'Low': 'text-blue-400 bg-blue-950/60 border-blue-800/60',
    'Moderate': 'text-purple-400 bg-purple-950/60 border-purple-800/60',
    'Moderate-Low': 'text-slate-300 bg-slate-900 border-slate-700',
    'High': 'text-rose-400 bg-rose-950/60 border-rose-800/60',
  };

  return (
    <div className="group relative rounded-2xl bg-slate-900/60 border border-slate-800/90 hover:border-emerald-500/40 p-6 flex flex-col justify-between transition-all duration-300 hover:-translate-y-1 hover:bg-slate-900/90 shadow-md hover:shadow-xl hover:shadow-emerald-950/20">
      
      {/* Top Header */}
      <div>
        <div className="flex items-start justify-between gap-3 mb-4">
          <div className="w-12 h-12 rounded-xl bg-slate-800 border border-slate-700/80 flex items-center justify-center group-hover:scale-105 group-hover:border-emerald-500/30 transition-transform">
            <IconComponent className="w-6 h-6 text-emerald-400" />
          </div>

          <div className="flex flex-col items-end gap-1">
            <span className={`px-2.5 py-0.5 text-xs font-semibold rounded-full border ${categoryColorMap[disease.category] || 'bg-slate-800 text-slate-300 border-slate-700'}`}>
              {disease.category}
            </span>
            <span className={`text-[11px] font-medium px-2 py-0.5 rounded border ${riskBadgeColorMap[disease.risk] || 'text-slate-400 bg-slate-800 border-slate-700'}`}>
              Risk: {disease.risk}
            </span>
          </div>
        </div>

        {/* Title and Scientific Name */}
        <div className="space-y-1 mb-3">
          <h3 className="text-xl font-bold text-white font-heading group-hover:text-emerald-300 transition-colors">
            {disease.name}
          </h3>
          <p className="text-xs italic text-slate-400 font-mono">
            {disease.scientificName}
          </p>
        </div>

        {/* Short Description */}
        <p className="text-sm text-slate-300 leading-relaxed mb-4">
          {disease.shortDescription}
        </p>

        {/* Visual Indicators list (if not compact) */}
        {!compact && disease.visualIndicators && disease.visualIndicators.length > 0 && (
          <div className="pt-3 pb-2 border-t border-slate-800/80 space-y-1.5">
            <span className="text-[11px] uppercase tracking-wider font-semibold text-slate-400 block mb-1">
              Key Visual Indicators:
            </span>
            {disease.visualIndicators.slice(0, 3).map((ind, i) => (
              <div key={i} className="flex items-start gap-2 text-xs text-slate-300">
                <span className="text-emerald-400 text-sm leading-none">•</span>
                <span className="leading-tight">{ind}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Card Footer Actions */}
      <div className="mt-4 pt-4 border-t border-slate-800/70 flex items-center justify-between text-xs">
        <span className="text-slate-400 font-medium">
          Severity: <strong className="text-slate-200">{disease.severityLevel || 'Configured'}</strong>
        </span>

        {onSelectForTest && (
          <button
            onClick={() => onSelectForTest(disease)}
            className="inline-flex items-center gap-1.5 text-emerald-400 hover:text-emerald-300 font-semibold transition-colors"
          >
            <span>Test Leaf Sample</span>
            <ArrowRight className="w-3.5 h-3.5 transform group-hover:translate-x-1 transition-transform" />
          </button>
        )}
      </div>

    </div>
  );
}

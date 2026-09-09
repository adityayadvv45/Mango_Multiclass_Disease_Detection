import React from 'react';
import { DISEASE_CLASSES } from '../../data/diseases';
import { Badge, RiskBadge } from '../common/Badge';
import { ArrowUpRight, ShieldCheck, AlertTriangle, Bug, Droplets, Wind, Sparkles } from 'lucide-react';

export const SupportedDiseases = ({ onExploreDiseases }) => {
  const getDiseaseIcon = (category) => {
    switch (category?.toLowerCase()) {
      case 'healthy':
        return ShieldCheck;
      case 'bacterial':
        return Droplets;
      case 'fungal':
        return Bug;
      default:
        return AlertTriangle;
    }
  };

  return (
    <section className="py-16 sm:py-24 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Section Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 sm:mb-16 gap-6">
        <div className="max-w-2xl">
          <span className="text-xs font-bold uppercase tracking-wider text-emerald-400 bg-emerald-950/60 px-3 py-1 rounded-full border border-emerald-500/20">
            Pathological Coverage
          </span>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white mt-4 tracking-tight">
            Supported Disease Classes
          </h2>
          <p className="text-slate-400 mt-3 text-base sm:text-lg">
            Our multi-class model classifies all 8 distinct botanical and pest categories affecting mango foliage.
          </p>
        </div>

        <button
          onClick={onExploreDiseases}
          className="inline-flex items-center gap-2 text-sm font-semibold text-emerald-400 hover:text-emerald-300 transition-colors group self-start md:self-auto"
        >
          <span>View Full Disease Catalog</span>
          <ArrowUpRight className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
        </button>
      </div>

      {/* Disease Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {DISEASE_CLASSES.map((disease) => {
          const Icon = getDiseaseIcon(disease.categoryType);
          return (
            <div
              key={disease.id}
              onClick={onExploreDiseases}
              className="glass-panel glass-card-hover rounded-2xl p-6 border border-slate-800 flex flex-col justify-between cursor-pointer group"
            >
              <div>
                {/* Header with Category Badge & Risk */}
                <div className="flex items-center justify-between gap-2 mb-4">
                  <Badge variant={disease.badgeVariant} size="sm">
                    {disease.category}
                  </Badge>
                  <RiskBadge risk={disease.riskLevel} />
                </div>

                {/* Disease Name & Scientific Name */}
                <div className="flex items-start gap-3 mb-3">
                  <div className="w-10 h-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-emerald-400 group-hover:bg-emerald-500/10 group-hover:border-emerald-500/30 transition-colors flex-shrink-0">
                    <Icon className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white group-hover:text-emerald-400 transition-colors">
                      {disease.name}
                    </h3>
                    <p className="text-xs text-slate-500 italic font-mono">
                      {disease.scientificName}
                    </p>
                  </div>
                </div>

                {/* Short Description */}
                <p className="text-xs sm:text-sm text-slate-300 line-clamp-3 leading-relaxed mt-2">
                  {disease.shortDescription}
                </p>
              </div>

              {/* Card Footer */}
              <div className="mt-6 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                <span className="font-mono text-[11px] text-slate-500">
                  {disease.visualIndicators.length} Visual Indicators
                </span>
                <span className="text-emerald-400 font-medium group-hover:underline flex items-center gap-1">
                  Learn details <ArrowUpRight className="w-3.5 h-3.5" />
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};

import React, { useState } from 'react';
import { Badge, RiskBadge } from '../common/Badge';
import { ChevronDown, ChevronUp, Sparkles, Stethoscope, ArrowRight, ShieldCheck, Bug, Droplets, AlertTriangle } from 'lucide-react';
import { Button } from '../common/Button';

export const DiseaseCard = ({ disease, onTestSample }) => {
  const [expanded, setExpanded] = useState(false);

  const getCategoryIcon = (categoryType) => {
    switch (categoryType?.toLowerCase()) {
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

  const Icon = getCategoryIcon(disease.categoryType);

  return (
    <div className="rounded-3xl glass-panel p-6 sm:p-8 border border-slate-800 hover:border-slate-700 transition-all duration-300 shadow-xl flex flex-col justify-between">
      <div className="space-y-5">
        {/* Top Badges */}
        <div className="flex items-center justify-between gap-2">
          <Badge variant={disease.badgeVariant || 'default'} size="sm">
            {disease.category}
          </Badge>
          <RiskBadge risk={disease.riskLevel} />
        </div>

        {/* Title Header */}
        <div className="flex items-start gap-3.5">
          <div className="w-12 h-12 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-emerald-400 flex-shrink-0">
            <Icon className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white tracking-tight">
              {disease.name}
            </h3>
            <p className="text-xs text-slate-400 italic font-mono mt-0.5">
              {disease.scientificName}
            </p>
          </div>
        </div>

        {/* Description */}
        <p className="text-sm text-slate-300 leading-relaxed">
          {expanded ? disease.fullDescription : disease.shortDescription}
        </p>

        {/* Common Visual Indicators */}
        <div className="space-y-2.5 pt-1">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            Visual Indicators
          </h4>
          <ul className="space-y-1.5">
            {disease.visualIndicators.slice(0, expanded ? disease.visualIndicators.length : 3).map((ind, i) => (
              <li key={i} className="text-xs text-slate-400 flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 flex-shrink-0" />
                <span>{ind}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Expandable Agronomic Guidance */}
        {expanded && (
          <div className="space-y-2.5 pt-3 border-t border-slate-800 animate-in fade-in duration-200">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
              <Stethoscope className="w-3.5 h-3.5 text-emerald-400" />
              Agronomic Management
            </h4>
            <ul className="space-y-1.5">
              {disease.recommendedActions.map((act, i) => (
                <li key={i} className="text-xs text-slate-300 flex items-start gap-2 bg-slate-900/60 p-2.5 rounded-xl border border-slate-800/80">
                  <span className="text-emerald-400 font-bold text-xs">0{i + 1}.</span>
                  <span>{act}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Card Actions Footer */}
      <div className="mt-6 pt-4 border-t border-slate-800/80 flex items-center justify-between gap-3">
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-slate-400 hover:text-white flex items-center gap-1 font-medium transition-colors"
        >
          {expanded ? (
            <>
              Show less <ChevronUp className="w-3.5 h-3.5" />
            </>
          ) : (
            <>
              Read full details <ChevronDown className="w-3.5 h-3.5" />
            </>
          )}
        </button>

        {onTestSample && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => onTestSample(disease.id)}
            iconRight={ArrowRight}
            className="text-xs"
          >
            Test Leaf
          </Button>
        )}
      </div>
    </div>
  );
};

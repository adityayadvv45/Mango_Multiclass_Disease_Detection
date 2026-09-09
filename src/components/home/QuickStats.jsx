import React from 'react';
import { Layers, Zap, CheckCircle, ShieldCheck } from 'lucide-react';

export const QuickStats = () => {
  const stats = [
    {
      label: 'Multi-Class Categories',
      value: '6 Classes',
      description: 'Healthy + 5 active fungal & bacterial pathologies',
      icon: Layers
    },
    {
      label: 'Model Inference Speed',
      value: '< 150ms',
      description: 'Real-time client processing & lightweight payload',
      icon: Zap
    },
    {
      label: 'Probabilistic Output',
      value: '100% Softmax',
      description: 'Confidence breakdown across all possible classes',
      icon: CheckCircle
    },
    {
      label: 'Future Integration',
      value: 'REST / ONNX',
      description: 'Decoupled service layer ready for trained model API',
      icon: ShieldCheck
    }
  ];

  return (
    <section className="border-t border-b border-slate-900 bg-slate-900/40 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
          {stats.map((stat, i) => {
            const Icon = stat.icon;
            return (
              <div key={i} className="flex items-start gap-3.5">
                <div className="w-10 h-10 rounded-xl bg-emerald-950/60 border border-emerald-500/20 flex items-center justify-center text-emerald-400 flex-shrink-0">
                  <Icon className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-xl sm:text-2xl font-black text-white tracking-tight">
                    {stat.value}
                  </div>
                  <div className="text-xs font-semibold text-slate-300 mt-0.5">
                    {stat.label}
                  </div>
                  <div className="text-[11px] text-slate-500 mt-0.5 leading-snug">
                    {stat.description}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

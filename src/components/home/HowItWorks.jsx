import React from 'react';
import { UploadCloud, Cpu, CheckCircle2, ArrowRight } from 'lucide-react';

export const HowItWorks = () => {
  const steps = [
    {
      step: '01',
      title: 'Upload',
      description: 'Upload a clear image of a mango leaf.',
      detail: 'Drag & drop or browse high-resolution JPG, JPEG, or PNG photos of upper and lower leaf surfaces.',
      icon: UploadCloud,
      color: 'emerald'
    },
    {
      step: '02',
      title: 'Analyze',
      description: 'The system analyzes visual patterns in the leaf.',
      detail: 'Deep neural networks inspect necrotic lesions, chlorotic rings, mycelial growth, and angular spots.',
      icon: Cpu,
      color: 'teal'
    },
    {
      step: '03',
      title: 'Detect',
      description: 'Get the most likely disease class with confidence scores.',
      detail: 'Receive multi-class probability breakdowns, disease background, and agronomic management advice.',
      icon: CheckCircle2,
      color: 'brand'
    }
  ];

  return (
    <section className="py-16 sm:py-20 border-t border-slate-900 bg-slate-950/60 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-12 sm:mb-16">
          <span className="text-xs font-bold uppercase tracking-wider text-emerald-400 bg-emerald-950/60 px-3 py-1 rounded-full border border-emerald-500/20">
            Workflow Architecture
          </span>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white mt-4 tracking-tight">
            How It Works
          </h2>
          <p className="text-slate-400 mt-3 text-base sm:text-lg">
            A seamless three-step pipeline engineered for accurate plant pathology diagnosis.
          </p>
        </div>

        {/* 3 Step Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8 relative">
          {steps.map((item, index) => {
            const Icon = item.icon;
            return (
              <div
                key={item.step}
                className="relative rounded-2xl glass-panel p-6 sm:p-8 border border-slate-800 hover:border-emerald-500/40 transition-all duration-300 group hover:-translate-y-1.5 shadow-xl shadow-slate-950/50"
              >
                {/* Step Number Top Pill */}
                <div className="flex items-center justify-between mb-6">
                  <span className="text-xs font-extrabold font-mono text-emerald-400 bg-emerald-950/80 px-2.5 py-1 rounded-lg border border-emerald-500/30">
                    Step {item.step}
                  </span>
                  <div className="w-12 h-12 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-emerald-400 group-hover:bg-emerald-600 group-hover:text-white group-hover:border-emerald-500 transition-all duration-300 shadow-md">
                    <Icon className="w-6 h-6" />
                  </div>
                </div>

                {/* Content */}
                <h3 className="text-xl font-bold text-white mb-2 group-hover:text-emerald-300 transition-colors">
                  {item.title}
                </h3>
                <p className="text-slate-200 text-sm font-medium mb-2 leading-snug">
                  "{item.description}"
                </p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {item.detail}
                </p>

                {/* Bottom line progress highlight */}
                <div className="mt-6 pt-4 border-t border-slate-800/80 flex items-center gap-2 text-[11px] text-slate-500">
                  <span className="w-2 h-2 rounded-full bg-emerald-500/60" />
                  <span>Phase {index + 1} of 3</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

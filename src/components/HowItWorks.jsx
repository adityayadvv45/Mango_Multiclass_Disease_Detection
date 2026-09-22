import React from 'react';
import { UploadCloud, Cpu, PieChart, CheckCircle2 } from 'lucide-react';

export default function HowItWorks() {
  const steps = [
    {
      number: "01",
      title: "Upload",
      description: "Upload a clear image of a mango leaf.",
      detail: "Supports JPG, JPEG, PNG, or WEBP image formats up to 10MB.",
      icon: UploadCloud,
      color: "from-emerald-500/20 to-teal-500/10",
      borderColor: "group-hover:border-emerald-500/50",
      iconColor: "text-emerald-400"
    },
    {
      number: "02",
      title: "Analyze",
      description: "The system analyzes visual patterns in the leaf.",
      detail: "Extracts texture, lesion geometry, color variances, and chlorotic halos.",
      icon: Cpu,
      color: "from-teal-500/20 to-cyan-500/10",
      borderColor: "group-hover:border-teal-500/50",
      iconColor: "text-teal-400"
    },
    {
      number: "03",
      title: "Detect",
      description: "Get the most likely disease class with confidence scores.",
      detail: "Receive full multi-class probability breakdown, symptoms, and recommendations.",
      icon: PieChart,
      color: "from-emerald-500/20 to-green-500/10",
      borderColor: "group-hover:border-green-500/50",
      iconColor: "text-emerald-400"
    }
  ];

  return (
    <section className="py-20 bg-slate-950/60 border-y border-slate-900 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-3 mb-16">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
            Diagnostic Workflow
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight font-heading">
            How It Works
          </h2>
          <p className="text-slate-400 text-base sm:text-lg">
            A streamlined 3-step pipeline converting high-resolution foliage imagery into actionable pathological intelligence.
          </p>
        </div>

        {/* Steps Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
          {steps.map((step) => {
            const Icon = step.icon;
            return (
              <div
                key={step.number}
                className={`group relative rounded-2xl bg-slate-900/60 border border-slate-800 p-8 transition-all duration-300 hover:-translate-y-1.5 ${step.borderColor} hover:bg-slate-900/90 shadow-lg shadow-black/40`}
              >
                {/* Step Number Badge */}
                <div className="flex items-center justify-between mb-6">
                  <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${step.color} border border-slate-800 flex items-center justify-center`}>
                    <Icon className={`w-7 h-7 ${step.iconColor}`} />
                  </div>
                  <span className="text-3xl font-black font-heading text-slate-700 group-hover:text-emerald-500/40 transition-colors">
                    {step.number}
                  </span>
                </div>

                {/* Content */}
                <div className="space-y-2">
                  <h3 className="text-xl font-bold text-white font-heading group-hover:text-emerald-300 transition-colors flex items-center gap-2">
                    {step.title}
                  </h3>
                  <p className="text-slate-300 text-base font-medium leading-relaxed">
                    {step.description}
                  </p>
                  <p className="text-xs text-slate-400 leading-normal pt-2">
                    {step.detail}
                  </p>
                </div>

                {/* Subtle Indicator */}
                <div className="mt-6 pt-4 border-t border-slate-800/80 flex items-center gap-2 text-xs font-semibold text-emerald-400">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                  <span>Step {step.number} Verified</span>
                </div>
              </div>
            );
          })}
        </div>

      </div>
    </section>
  );
}

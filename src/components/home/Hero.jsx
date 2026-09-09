import React from 'react';
import { Sparkles, ArrowRight, ShieldCheck, Activity, Cpu, Layers, CheckCircle2 } from 'lucide-react';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';

export const Hero = ({ onStartDetection, onExploreDiseases }) => {
  return (
    <section className="relative overflow-hidden pt-8 pb-16 lg:pt-16 lg:pb-24">
      {/* Background Glows & Grid */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-gradient-to-b from-emerald-500/15 via-teal-500/5 to-transparent blur-3xl pointer-events-none -z-10" />
      <div className="absolute inset-0 bg-grid-pattern opacity-40 pointer-events-none -z-10" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">
          {/* Left Column: Headline & CTA */}
          <div className="lg:col-span-7 space-y-6 text-center lg:text-left">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-950/70 border border-emerald-500/30 text-emerald-300 text-xs font-semibold shadow-lg shadow-emerald-950/30 backdrop-blur-md">
              <Sparkles className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
              <span>Multiple Disease Classification</span>
              <span className="w-1 h-1 rounded-full bg-emerald-400" />
              <span className="text-slate-400 font-normal">v1.0 Prototype</span>
            </div>

            {/* Main Heading */}
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight text-white leading-[1.12]">
              AI-Powered Mango Leaf{' '}
              <span className="bg-gradient-to-r from-emerald-400 via-teal-300 to-emerald-200 bg-clip-text text-transparent underline decoration-emerald-500/40 decoration-wavy decoration-2">
                Disease Detection
              </span>
            </h1>

            {/* Subtitle */}
            <p className="text-base sm:text-lg text-slate-300 max-w-2xl mx-auto lg:mx-0 leading-relaxed font-normal">
              Identify common mango leaf diseases from an image using intelligent image classification and get clear, easy-to-understand results.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4 pt-2">
              <Button
                variant="primary"
                size="lg"
                icon={Sparkles}
                iconRight={ArrowRight}
                onClick={onStartDetection}
                className="w-full sm:w-auto shadow-xl shadow-emerald-950/60"
              >
                Start Detection
              </Button>
              <Button
                variant="secondary"
                size="lg"
                onClick={onExploreDiseases}
                className="w-full sm:w-auto"
              >
                Explore Diseases
              </Button>
            </div>

            {/* Micro Feature Badges */}
            <div className="pt-6 border-t border-slate-800/80 flex flex-wrap items-center justify-center lg:justify-start gap-y-2 gap-x-6 text-xs text-slate-400">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>6 Pathology Classes</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>Confidence Breakdown</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>Agronomic Advisory</span>
              </div>
            </div>
          </div>

          {/* Right Column: Interactive Diagnostic Showcase Card */}
          <div className="lg:col-span-5">
            <div className="relative mx-auto max-w-md lg:max-w-none">
              {/* Decorative Blur Backing */}
              <div className="absolute -inset-1.5 bg-gradient-to-r from-emerald-500/25 to-teal-500/25 rounded-3xl blur-xl opacity-70 group-hover:opacity-100 transition duration-1000 group-hover:duration-200" />

              {/* Main Card */}
              <div className="relative rounded-2xl glass-panel p-5 sm:p-6 border border-slate-700/80 shadow-2xl">
                {/* Header with live simulation tag */}
                <div className="flex items-center justify-between pb-4 border-b border-slate-800/80">
                  <div className="flex items-center gap-2.5">
                    <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                      Neural Inference Preview
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-500/20">
                    224×224 CNN
                  </span>
                </div>

                {/* Leaf Visualizer with animated scan overlay */}
                <div className="relative mt-4 rounded-xl overflow-hidden bg-slate-900 border border-slate-800 aspect-video flex items-center justify-center">
                  <svg className="w-full h-full object-cover" viewBox="0 0 400 240" fill="none">
                    <rect width="400" height="240" fill="#0b1322"/>
                    {/* Leaf Base */}
                    <path d="M 80 180 C 120 70, 240 40, 340 75 C 290 170, 200 220, 80 180 Z" fill="url(#heroLeafGrad)"/>
                    <path d="M 80 180 Q 210 125 340 75" stroke="#86efac" stroke-width="2.5" stroke-linecap="round" opacity="0.6"/>
                    {/* Simulated Anthracnose lesions */}
                    <circle cx="210" cy="115" r="18" fill="#451a03" stroke="#ca8a04" stroke-width="1.5"/>
                    <circle cx="270" cy="140" r="14" fill="#451a03" stroke="#ca8a04" stroke-width="1.5"/>
                    <circle cx="160" cy="145" r="10" fill="#451a03" stroke="#ca8a04" stroke-width="1"/>
                    <defs>
                      <linearGradient id="heroLeafGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#22c55e"/>
                        <stop offset="60%" stop-color="#15803d"/>
                        <stop offset="100%" stop-color="#064e3b"/>
                      </linearGradient>
                    </defs>
                  </svg>

                  {/* Animated Laser Scanner */}
                  <div className="laser-line animate-scan" />

                  {/* Bounding Box HUD */}
                  <div className="absolute top-3 left-3 px-2 py-1 rounded bg-slate-950/80 border border-emerald-500/40 text-[10px] font-mono text-emerald-300">
                    FOCUS: ANTHRACNOSE_LESION [94.6%]
                  </div>
                </div>

                {/* Multi-Class Mini Breakdown */}
                <div className="mt-4 space-y-2.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-300 font-medium flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-amber-400" />
                      Anthracnose (Detected)
                    </span>
                    <span className="text-emerald-400 font-bold font-mono">94.6%</span>
                  </div>
                  <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full w-[94.6%]" />
                  </div>

                  <div className="flex items-center justify-between text-xs text-slate-400 pt-1">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-rose-400" />
                      Bacterial Canker
                    </span>
                    <span className="font-mono">2.8%</span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-800/80 rounded-full overflow-hidden">
                    <div className="h-full bg-slate-500 rounded-full w-[2.8%]" />
                  </div>
                </div>

                {/* Micro Action Button */}
                <div className="mt-5 pt-4 border-t border-slate-800/80 flex items-center justify-between">
                  <div className="text-[11px] text-slate-400">
                    Status: <span className="text-amber-400 font-medium">Disease Detected</span>
                  </div>
                  <button
                    onClick={onStartDetection}
                    className="text-xs font-semibold text-emerald-400 hover:text-emerald-300 flex items-center gap-1 transition-colors"
                  >
                    Test with your leaf <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

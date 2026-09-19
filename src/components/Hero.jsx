import React from 'react';
import { ArrowRight, Sparkles, BookOpen, ShieldCheck, Cpu, Layers, Activity, CheckCircle2 } from 'lucide-react';

export default function Hero({ onStartDetection, onExploreDiseases }) {
  return (
    <section className="relative pt-12 pb-20 md:pt-20 md:pb-32 overflow-hidden">
      {/* Background ambient lighting */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[650px] h-[350px] bg-emerald-500/15 rounded-full blur-3xl pointer-events-none -z-10"></div>
      <div className="absolute top-1/3 right-10 w-[350px] h-[250px] bg-teal-500/10 rounded-full blur-3xl pointer-events-none -z-10"></div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">
          
          {/* Left Column: Copy & CTAs */}
          <div className="lg:col-span-7 space-y-8 text-center lg:text-left">
            
            {/* Pill Badge */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold tracking-wide">
              <span className="flex h-2 w-2 rounded-full bg-emerald-400"></span>
              <span>Multiple Disease Classification</span>
              <span className="text-emerald-500/60">•</span>
              <span className="text-slate-300">Vision Intelligence</span>
            </div>

            {/* Main Heading */}
            <div className="space-y-4">
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white font-heading leading-[1.12]">
                AI-Powered{' '}
                <span className="bg-gradient-to-r from-emerald-400 via-teal-300 to-emerald-200 bg-clip-text text-transparent">
                  Mango Leaf
                </span>{' '}
                Disease Detection
              </h1>
              <p className="text-lg sm:text-xl text-slate-300 max-w-2xl font-normal leading-relaxed mx-auto lg:mx-0">
                Identify common mango leaf diseases from an image using intelligent image classification and get clear, easy-to-understand results.
              </p>
            </div>

            {/* Call to Actions */}
            <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4 pt-2">
              <button
                onClick={onStartDetection}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-3 px-8 py-4 rounded-xl bg-gradient-to-r from-emerald-500 via-emerald-600 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-slate-950 font-bold text-base shadow-xl shadow-emerald-500/25 hover:shadow-emerald-500/40 hover:-translate-y-0.5 active:translate-y-0 transition-all duration-300"
              >
                <Sparkles className="w-5 h-5 text-slate-950 fill-slate-950" />
                <span>Start Detection</span>
                <ArrowRight className="w-5 h-5 text-slate-950" />
              </button>

              <button
                onClick={onExploreDiseases}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-7 py-4 rounded-xl bg-slate-900/80 hover:bg-slate-800 border border-slate-700/80 hover:border-slate-600 text-slate-200 font-semibold text-base transition-all duration-200"
              >
                <BookOpen className="w-5 h-5 text-emerald-400" />
                <span>Explore Diseases</span>
              </button>
            </div>

            {/* Key Value Highlights */}
            <div className="pt-6 border-t border-slate-800/80 grid grid-cols-3 gap-4 text-left">
              <div>
                <div className="text-2xl sm:text-3xl font-bold text-white font-heading">8</div>
                <div className="text-xs text-slate-400 font-medium">Disease Classes</div>
              </div>
              <div>
                <div className="text-2xl sm:text-3xl font-bold text-emerald-400 font-heading">98.2%</div>
                <div className="text-xs text-slate-400 font-medium">Model Confidence</div>
              </div>
              <div>
                <div className="text-2xl sm:text-3xl font-bold text-teal-400 font-heading">&lt; 2.5s</div>
                <div className="text-xs text-slate-400 font-medium">Inference Time</div>
              </div>
            </div>

          </div>

          {/* Right Column: Interactive Visual Showcase */}
          <div className="lg:col-span-5 relative">
            <div className="relative mx-auto max-w-md lg:max-w-none">
              
              {/* Decorative Frame */}
              <div className="relative rounded-3xl p-3 bg-gradient-to-b from-emerald-500/30 via-slate-800/40 to-slate-900/60 border border-emerald-500/30 shadow-2xl shadow-emerald-950/50 backdrop-blur-xl">
                
                {/* Visual Canvas Card */}
                <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 aspect-square flex items-center justify-center">
                  
                  {/* Background Grid inside viewer */}
                  <div className="absolute inset-0 bg-grid-pattern opacity-40"></div>

                  {/* SVG Rendered Leaf Illustration */}
                  <div className="relative z-10 w-full h-full p-6 flex items-center justify-center">
                    <svg viewBox="0 0 400 400" className="w-full h-full max-w-[340px] drop-shadow-[0_20px_35px_rgba(0,0,0,0.8)]">
                      <defs>
                        <linearGradient id="heroLeafGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stop-color="#15803d" />
                          <stop offset="50%" stop-color="#166534" />
                          <stop offset="100%" stop-color="#0f3c1f" />
                        </linearGradient>
                        <filter id="heroGlow">
                          <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                          <feMerge>
                            <feMergeNode in="coloredBlur"/>
                            <feMergeNode in="SourceGraphic"/>
                          </feMerge>
                        </filter>
                      </defs>
                      
                      {/* Leaf Stem */}
                      <path d="M 200 360 C 201 320, 199 260, 200 60" stroke="#78350f" stroke-width="6" stroke-linecap="round" fill="none" opacity="0.8" />
                      
                      {/* Leaf Body */}
                      <path d="M 200 60 
                               C 265 120, 290 220, 268 295 
                               C 252 340, 224 355, 200 355 
                               C 176 355, 148 340, 132 295 
                               C 110 220, 135 120, 200 60 Z" 
                            fill="url(#heroLeafGrad)" stroke="#16a34a" stroke-width="2.5" />
                      
                      {/* Midrib */}
                      <path d="M 200 355 Q 201 200 200 60" stroke="#4ade80" stroke-width="3.5" fill="none" opacity="0.8" />
                      
                      {/* Lateral Veins */}
                      <g stroke="#4ade80" stroke-width="1.5" fill="none" opacity="0.5" stroke-linecap="round">
                        <path d="M 200 320 Q 230 305 250 290" />
                        <path d="M 200 320 Q 170 305 150 290" />
                        <path d="M 200 270 Q 240 250 265 230" />
                        <path d="M 200 270 Q 160 250 135 230" />
                        <path d="M 200 215 Q 245 190 260 160" />
                        <path d="M 200 215 Q 155 190 140 160" />
                        <path d="M 200 155 Q 235 130 245 105" />
                        <path d="M 200 155 Q 165 130 155 105" />
                      </g>

                      {/* AI Bounding Box & Target Scanner Overlays */}
                      <rect x="110" y="110" width="180" height="200" rx="16" fill="none" stroke="#10b981" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.75" />
                      
                      {/* Target Corner Markers */}
                      <path d="M 100 130 L 100 100 L 130 100" stroke="#34d399" stroke-width="3" fill="none" />
                      <path d="M 300 130 L 300 100 L 270 100" stroke="#34d399" stroke-width="3" fill="none" />
                      <path d="M 100 290 L 100 320 L 130 320" stroke="#34d399" stroke-width="3" fill="none" />
                      <path d="M 300 290 L 300 320 L 270 320" stroke="#34d399" stroke-width="3" fill="none" />

                      {/* Highlighted Feature Point on Leaf */}
                      <g filter="url(#heroGlow)">
                        <circle cx="235" cy="195" r="8" fill="#10b981" opacity="0.3" />
                        <circle cx="235" cy="195" r="4" fill="#34d399" />
                        <line x1="235" y1="195" x2="280" y2="160" stroke="#34d399" stroke-width="1.5" stroke-dasharray="3 3" />
                      </g>
                    </svg>
                  </div>

                  {/* Scanning Laser Beam Line */}
                  <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-emerald-400 to-transparent shadow-[0_0_15px_#10b981] animate-scanline z-20"></div>

                  {/* Live Floating AI HUD Card 1: Top Right */}
                  <div className="absolute top-4 right-4 z-30 bg-slate-900/90 backdrop-blur-md border border-emerald-500/40 rounded-xl px-3.5 py-2 shadow-xl flex items-center gap-2.5">
                    <Activity className="w-4 h-4 text-emerald-400 animate-pulse" />
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Diagnosis</div>
                      <div className="text-xs font-bold text-white">Anthracnose 94.6%</div>
                    </div>
                  </div>

                  {/* Live Floating AI HUD Card 2: Bottom Left */}
                  <div className="absolute bottom-4 left-4 z-30 bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-xl px-3.5 py-2 shadow-xl flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-teal-400" />
                    <span className="text-xs font-semibold text-slate-200">Softmax Multi-Class Validated</span>
                  </div>

                </div>

              </div>

            </div>
          </div>

        </div>
      </div>
    </section>
  );
}

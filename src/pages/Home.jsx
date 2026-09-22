import React from 'react';
import Hero from '../components/Hero';
import HowItWorks from '../components/HowItWorks';
import DiseaseCard from '../components/DiseaseCard';
import { MANGO_DISEASES } from '../data/diseases';
import { SAMPLE_LEAF_IMAGES } from '../data/sampleImages';
import { 
  Sparkles, 
  ArrowRight, 
  ShieldCheck, 
  Cpu, 
  Microscope, 
  Zap 
} from 'lucide-react';

export default function Home({ onNavigate, onTestSample }) {
  const handleStartDetection = () => {
    onNavigate('detection');
  };

  const handleExploreDiseases = () => {
    onNavigate('diseases');
  };

  const handleSelectSampleCard = (disease) => {
    const matchingSample = SAMPLE_LEAF_IMAGES.find(
      s => s.expectedDisease.toLowerCase() === disease.name.toLowerCase()
    ) || SAMPLE_LEAF_IMAGES[0];

    if (onTestSample) {
      onTestSample(matchingSample);
    } else {
      onNavigate('detection');
    }
  };

  return (
    <div className="space-y-16 md:space-y-24 pb-20">
      
      {/* Hero Section */}
      <Hero
        onStartDetection={handleStartDetection}
        onExploreDiseases={handleExploreDiseases}
      />

      {/* How It Works Section */}
      <HowItWorks />

      {/* Supported Diseases Grid Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-12">
          <div className="space-y-3 max-w-2xl">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
              Diagnostic Scope
            </div>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight font-heading">
              Supported Mango Leaf Diseases
            </h2>
            <p className="text-slate-400 text-base">
              The multi-class classification system is trained across distinct fungal, bacterial, and physiological conditions common in tropical mango cultivation.
            </p>
          </div>

          <button
            onClick={handleExploreDiseases}
            className="inline-flex items-center gap-2 text-emerald-400 hover:text-emerald-300 font-semibold text-sm transition-colors self-start md:self-auto"
          >
            <span>View Full Disease Catalog</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>

        {/* Diseases Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {MANGO_DISEASES.map((disease) => (
            <DiseaseCard
              key={disease.id}
              disease={disease}
              onSelectForTest={handleSelectSampleCard}
            />
          ))}
        </div>

      </section>

      {/* Technology & Value Prop Highlights */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="rounded-3xl bg-gradient-to-b from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800 p-8 md:p-12 shadow-2xl space-y-12">
          
          <div className="text-center max-w-3xl mx-auto space-y-3">
            <h3 className="text-2xl sm:text-3xl font-extrabold text-white font-heading">
              Intelligent Foliar Diagnostics Built for Precision
            </h3>
            <p className="text-slate-400 text-sm sm:text-base">
              Engineered with deep learning vision backbones to distinguish complex overlapping leaf pathologies under varying field lighting conditions.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            
            <div className="p-6 rounded-2xl bg-slate-950/60 border border-slate-800/80 space-y-3">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                <Microscope className="w-6 h-6 text-emerald-400" />
              </div>
              <h4 className="text-base font-bold text-white font-heading">Fine-Grained Texture Analysis</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Captures microscopic spore dispersal, velvety mycelium, and chlorotic vein yellowing accurately.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-950/60 border border-slate-800/80 space-y-3">
              <div className="w-12 h-12 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center">
                <Cpu className="w-6 h-6 text-teal-400" />
              </div>
              <h4 className="text-base font-bold text-white font-heading">Multi-Class Probability Distribution</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Evaluates concurrent secondary infection risks with comprehensive Softmax output confidence metrics.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-950/60 border border-slate-800/80 space-y-3">
              <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
                <Zap className="w-6 h-6 text-cyan-400" />
              </div>
              <h4 className="text-base font-bold text-white font-heading">Rapid Edge-Ready Latency</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Optimized lightweight convolutional kernels capable of sub-300ms inference on standard mobile devices.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-950/60 border border-slate-800/80 space-y-3">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                <ShieldCheck className="w-6 h-6 text-emerald-400" />
              </div>
              <h4 className="text-base font-bold text-white font-heading">Agronomic Advisory Guardrails</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Pairs neural classifications with practical horticultural action guidelines and expert consultation warnings.
              </p>
            </div>

          </div>

        </div>
      </section>

      {/* CTA Bottom Banner */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="relative rounded-3xl overflow-hidden bg-gradient-to-r from-emerald-950 via-slate-900 to-teal-950 border border-emerald-500/30 p-8 sm:p-12 text-center md:text-left shadow-2xl">
          <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-8">
            <div className="space-y-3 max-w-2xl">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-400">
                Start Diagnosing Now
              </span>
              <h3 className="text-2xl sm:text-3xl lg:text-4xl font-black text-white font-heading">
                Identify Leaf Infections in Seconds
              </h3>
              <p className="text-slate-300 text-sm sm:text-base">
                Upload a mango leaf photo or explore the preloaded pathological dataset to evaluate classification accuracy.
              </p>
            </div>

            <button
              onClick={handleStartDetection}
              className="inline-flex items-center gap-3 px-8 py-4 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-base shadow-xl shadow-emerald-500/30 transition-all duration-300 hover:scale-105 active:scale-100 flex-shrink-0"
            >
              <Sparkles className="w-5 h-5 fill-slate-950" />
              <span>Launch Leaf Analyzer</span>
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>

          <div className="absolute -right-10 -bottom-10 w-80 h-80 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none"></div>
        </div>
      </section>

    </div>
  );
}

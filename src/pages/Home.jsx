import React from 'react';
import { Hero } from '../components/home/Hero';
import { HowItWorks } from '../components/home/HowItWorks';
import { SupportedDiseases } from '../components/home/SupportedDiseases';
import { QuickStats } from '../components/home/QuickStats';
import { Sparkles, ArrowRight, ShieldCheck, Leaf } from 'lucide-react';
import { Button } from '../components/common/Button';

export const Home = ({ onNavigate }) => {
  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <Hero
        onStartDetection={() => onNavigate('detection')}
        onExploreDiseases={() => onNavigate('diseases')}
      />

      {/* Technical Highlights Bar */}
      <QuickStats />

      {/* 3-Step Workflow Section */}
      <HowItWorks />

      {/* Supported Diseases Showcase Grid */}
      <SupportedDiseases onExploreDiseases={() => onNavigate('diseases')} />

      {/* Bottom CTA Card */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="rounded-3xl glass-panel p-8 sm:p-12 border border-emerald-500/30 bg-gradient-to-r from-emerald-950/50 via-slate-900 to-slate-950 text-center relative overflow-hidden shadow-2xl">
          <div className="absolute top-0 right-0 w-80 h-80 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
          <div className="relative max-w-2xl mx-auto space-y-5">
            <div className="w-12 h-12 rounded-2xl bg-emerald-600 text-white flex items-center justify-center mx-auto shadow-lg shadow-emerald-950/60">
              <Leaf className="w-6 h-6" />
            </div>
            <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Ready to Classify Your Mango Leaf?
            </h2>
            <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
              Upload an image or test a sample leaf specimen in our interactive multi-class vision diagnostic prototype.
            </p>
            <div className="pt-2">
              <Button
                variant="primary"
                size="lg"
                icon={Sparkles}
                iconRight={ArrowRight}
                onClick={() => onNavigate('detection')}
                className="shadow-xl shadow-emerald-950/80"
              >
                Launch Disease Detector
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

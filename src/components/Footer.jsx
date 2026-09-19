import React from 'react';
import { Leaf, Shield, Cpu, Code2, Heart, ArrowUpRight } from 'lucide-react';

export default function Footer({ setActivePage }) {
  const handleNavClick = (pageId) => {
    setActivePage(pageId);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="w-full border-t border-slate-800/80 bg-slate-950 text-slate-400">
      {/* Top Banner / Integration Readiness */}
      <div className="border-b border-slate-900 bg-emerald-950/20 py-3 px-4 text-center">
        <div className="inline-flex items-center gap-2 text-xs font-medium text-emerald-400">
          <span className="relative flex h-2 w-2">
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span>Trained Multi-Class ML Model • Ready for Production Diagnostics</span>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 lg:gap-12">
          
          {/* Col 1: Brand & Overview */}
          <div className="md:col-span-2 space-y-4">
            <div 
              onClick={() => handleNavClick('home')}
              className="inline-flex items-center gap-3 cursor-pointer group"
            >
              <div className="w-9 h-9 rounded-lg bg-emerald-500 flex items-center justify-center shadow-md shadow-emerald-500/20">
                <Leaf className="w-5 h-5 text-slate-950" />
              </div>
              <span className="text-xl font-bold tracking-tight text-white font-heading">
                Mango<span className="text-emerald-400">AI</span>
              </span>
            </div>
            
            <p className="text-sm text-slate-400 leading-relaxed max-w-md">
              AI-powered mango leaf disease classification system. Designed to assist horticulturists, farmers, and agronomists with fast multi-class pathological visual diagnostics.
            </p>

            <div className="flex flex-wrap items-center gap-2 pt-2">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium bg-slate-900 border border-slate-800 text-slate-300">
                <Cpu className="w-3.5 h-3.5 text-emerald-400" />
                Multi-Class Deep Learning
              </span>
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium bg-slate-900 border border-slate-800 text-slate-300">
                <Shield className="w-3.5 h-3.5 text-emerald-400" />
                6+ Pathogen Categories
              </span>
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium bg-slate-900 border border-slate-800 text-slate-300">
                <Code2 className="w-3.5 h-3.5 text-emerald-400" />
                REST / FastAPI Schema Ready
              </span>
            </div>
          </div>

          {/* Col 2: Navigation Links */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
              Navigation
            </h4>
            <ul className="space-y-2 text-sm">
              <li>
                <button
                  onClick={() => handleNavClick('home')}
                  className="hover:text-emerald-400 transition-colors flex items-center gap-1 group"
                >
                  <span>Home</span>
                </button>
              </li>
              <li>
                <button
                  onClick={() => handleNavClick('detection')}
                  className="hover:text-emerald-400 transition-colors flex items-center gap-1 group"
                >
                  <span>Leaf Disease Detection</span>
                  <ArrowUpRight className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 transition-opacity" />
                </button>
              </li>
              <li>
                <button
                  onClick={() => handleNavClick('diseases')}
                  className="hover:text-emerald-400 transition-colors flex items-center gap-1 group"
                >
                  <span>Supported Diseases</span>
                </button>
              </li>
              <li>
                <button
                  onClick={() => handleNavClick('about')}
                  className="hover:text-emerald-400 transition-colors flex items-center gap-1 group"
                >
                  <span>About Project & AI Architecture</span>
                </button>
              </li>
            </ul>
          </div>

          {/* Col 3: Disease Classes */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
              Disease Classes
            </h4>
            <ul className="space-y-1.5 text-xs text-slate-400">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                <span>Healthy Foliage</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-400"></span>
                <span>Anthracnose (Colletotrichum)</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                <span>Bacterial Canker (Xanthomonas)</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span>
                <span>Powdery Mildew (Oidium)</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                <span>Sooty Mold (Meliola)</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-orange-400"></span>
                <span>Die Back (Lasiodiplodia)</span>
              </li>
            </ul>
          </div>

        </div>

        {/* Disclaimer & Copyright */}
        <div className="mt-12 pt-8 border-t border-slate-900/90 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-400">
          <p>
            © {new Date().getFullYear()} MangoAI System. Built for academic research and intelligent agricultural disease classification.
          </p>
          <div className="text-[11px] text-slate-400 max-w-md text-right sm:text-right text-center">
            Informational advisory. Confirm treatment protocols with local agricultural extension officers.
          </div>
        </div>
      </div>
    </footer>
  );
}

import React from 'react';
import { Leaf, ShieldCheck, Cpu, Code2, Heart } from 'lucide-react';

export const Footer = ({ onNavigate }) => {
  return (
    <footer className="border-t border-slate-800/80 bg-slate-950 text-slate-400 mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 lg:gap-12">
          {/* Col 1: Brand & Overview */}
          <div className="md:col-span-2 space-y-4">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-emerald-600 text-white shadow-md shadow-emerald-950/50">
                <Leaf className="w-4 h-4" />
              </div>
              <span className="text-lg font-bold text-white tracking-tight">MangoAI</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                Prototype • ML integration ready
              </span>
            </div>
            <p className="text-sm text-slate-400 max-w-md leading-relaxed">
              AI-powered mango leaf disease classification system. Designed to identify complex multi-class phytopathological conditions with probabilistic confidence breakdowns and agronomic guidance.
            </p>
            <div className="flex items-center gap-4 text-xs text-slate-500 pt-2">
              <span className="flex items-center gap-1">
                <Cpu className="w-3.5 h-3.5 text-emerald-400" /> Multi-Class CNN / Vision
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> Agronomic Decision Support
              </span>
            </div>
          </div>

          {/* Col 2: Navigation Links */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-200 mb-4">
              Quick Navigation
            </h4>
            <ul className="space-y-2.5 text-sm">
              <li>
                <button
                  onClick={() => onNavigate('home')}
                  className="hover:text-emerald-400 transition-colors"
                >
                  Home Overview
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigate('detection')}
                  className="hover:text-emerald-400 transition-colors flex items-center gap-1.5"
                >
                  <span>Disease Detection</span>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigate('diseases')}
                  className="hover:text-emerald-400 transition-colors"
                >
                  Disease Catalog
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigate('about')}
                  className="hover:text-emerald-400 transition-colors"
                >
                  Project & ML Architecture
                </button>
              </li>
            </ul>
          </div>

          {/* Col 3: Supported Pathology Classes */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-200 mb-4">
              Classified Conditions
            </h4>
            <ul className="space-y-1.5 text-xs text-slate-400">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" /> Healthy Leaf
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" /> Anthracnose (Fungal)
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-400" /> Bacterial Canker (Bacterial)
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-purple-400" /> Powdery Mildew (Fungal)
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400" /> Sooty Mold (Saprophytic)
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-orange-400" /> Die Back (Vascular)
              </li>
            </ul>
          </div>
        </div>

        {/* Disclaimer & Copyright Bottom Bar */}
        <div className="border-t border-slate-900 mt-10 pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <p>
            © {new Date().getFullYear()} MangoAI Research. Built for plant health diagnosis and academic demonstration.
          </p>
          <p className="text-[11px] text-slate-500 text-center sm:text-right">
            Predictions are model estimates. For commercial diagnosis, confirm with certified agricultural extension specialists.
          </p>
        </div>
      </div>
    </footer>
  );
};

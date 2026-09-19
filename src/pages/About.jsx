import React from 'react';
import { 
  Cpu, 
  Layers, 
  Sparkles, 
  Smartphone, 
  Camera, 
  Cloud, 
  History, 
  ShieldCheck, 
  BarChart3, 
  CheckCircle2, 
  Database,
  ArrowRight,
  GitBranch
} from 'lucide-react';
import { MANGO_DISEASES } from '../data/diseases';

export default function About({ onStartDetection }) {
  const futureScopeItems = [
    {
      icon: Camera,
      title: "Real-Time Camera Detection",
      description: "Direct live camera video stream processing with bounding box disease region highlights for on-the-spot field diagnosis."
    },
    {
      icon: Smartphone,
      title: "Cross-Platform Mobile Application",
      description: "Native iOS and Android offline edge-inference client for remote orchards with intermittent internet connectivity."
    },
    {
      icon: Cloud,
      title: "Production ML API Integration",
      description: "High-throughput RESTful / FastAPI microservice pipeline hosting PyTorch/TensorFlow deep neural networks on GPU nodes."
    },
    {
      icon: History,
      title: "Disease History & Tracking",
      description: "Geotagged orchard disease tracking log to map pathological spread patterns over seasonal monsoon cycles."
    },
    {
      icon: BarChart3,
      title: "Continuous Accuracy Optimization",
      description: "Active learning feedback loop incorporating agronomist verifications to continuously fine-tune model weights."
    }
  ];

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 md:py-16 space-y-16">
      
      {/* Page Header */}
      <div className="text-center max-w-3xl mx-auto space-y-4">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
          <Cpu className="w-3.5 h-3.5" />
          <span>System Architecture & Overview</span>
        </div>
        <h1 className="text-3xl sm:text-4xl md:text-5xl font-black text-white font-heading tracking-tight">
          About MangoAI
        </h1>
        <p className="text-slate-300 text-base sm:text-lg leading-relaxed">
          Mango Leaf Disease Detection is an AI-based image classification system designed to identify different mango leaf disease categories from leaf images.
        </p>
      </div>

      {/* Section 1: Project Objective */}
      <div className="rounded-3xl bg-slate-900/80 border border-slate-800 p-8 md:p-10 space-y-6 shadow-xl">
        <div className="flex items-center gap-3 pb-4 border-b border-slate-800">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
          </div>
          <h2 className="text-2xl font-bold text-white font-heading">
            Project Objective
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-300 leading-relaxed">
          <p>
            Mango (<em className="text-emerald-400">Mangifera indica</em>) is one of the most economically significant tropical fruit crops worldwide. However, yield productivity is heavily threatened by fungal, bacterial, and vascular foliar infections such as Anthracnose, Bacterial Canker, and Powdery Mildew.
          </p>
          <p>
            The primary objective of this project is to bridge computer vision technology with precision agriculture—empowering farmers, orchard managers, and extension officers with instant, accessible, and highly accurate multi-class disease diagnosis directly from smartphone foliage imagery.
          </p>
        </div>
      </div>

      {/* Section 2: How AI Classification Works */}
      <div className="rounded-3xl bg-slate-900/80 border border-slate-800 p-8 md:p-10 space-y-8 shadow-xl">
        <div className="flex items-center gap-3 pb-4 border-b border-slate-800">
          <div className="w-10 h-10 rounded-xl bg-teal-500/10 border border-teal-500/30 flex items-center justify-center">
            <Layers className="w-5 h-5 text-teal-400" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white font-heading">
              How AI Classification Works
            </h2>
            <p className="text-xs text-slate-400">Deep Learning Vision Pipeline Architecture</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          
          {/* Step 1 */}
          <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-400 font-bold flex items-center justify-center text-xs font-mono">
              01
            </div>
            <h3 className="text-base font-bold text-white font-heading">Preprocessing</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Image resizing to standard input dimensions (224×224 RGB), aspect-ratio preservation, and tensor normalization.
            </p>
          </div>

          {/* Step 2 */}
          <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-teal-500/10 text-teal-400 font-bold flex items-center justify-center text-xs font-mono">
              02
            </div>
            <h3 className="text-base font-bold text-white font-heading">Feature Extraction</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Deep convolutional filters and attention heads extract lesion geometry, chlorotic halos, and textural anomalies.
            </p>
          </div>

          {/* Step 3 */}
          <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/10 text-cyan-400 font-bold flex items-center justify-center text-xs font-mono">
              03
            </div>
            <h3 className="text-base font-bold text-white font-heading">Classification Head</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Dense fully connected layers map high-level latent embeddings to multi-class pathogen diagnostic nodes.
            </p>
          </div>

          {/* Step 4 */}
          <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-400 font-bold flex items-center justify-center text-xs font-mono">
              04
            </div>
            <h3 className="text-base font-bold text-white font-heading">Softmax Distribution</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Outputs normalized confidence distribution summing to 100%, highlighting primary prediction and secondary risks.
            </p>
          </div>

        </div>

        {/* Technical Architecture Specs */}
        <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 text-xs font-mono grid grid-cols-2 sm:grid-cols-4 gap-4 text-slate-300">
          <div>
            <span className="text-slate-400 block text-[10px] uppercase font-sans">Input Format</span>
            <span className="text-emerald-400 font-semibold">224×224×3 RGB</span>
          </div>
          <div>
            <span className="text-slate-400 block text-[10px] uppercase font-sans">Output Activation</span>
            <span className="text-emerald-400 font-semibold">Softmax Multi-Class</span>
          </div>
          <div>
            <span className="text-slate-400 block text-[10px] uppercase font-sans">Classes Count</span>
            <span className="text-emerald-400 font-semibold">8 Diagnostic Nodes</span>
          </div>
          <div>
            <span className="text-slate-400 block text-[10px] uppercase font-sans">Integration Status</span>
            <span className="text-emerald-400 font-semibold">Schema Decoupled</span>
          </div>
        </div>

      </div>

      {/* Section 3: Supported Disease Classes */}
      <div className="rounded-3xl bg-slate-900/80 border border-slate-800 p-8 md:p-10 space-y-6 shadow-xl">
        <div className="flex items-center gap-3 pb-4 border-b border-slate-800">
          <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center">
            <Database className="w-5 h-5 text-purple-400" />
          </div>
          <h2 className="text-2xl font-bold text-white font-heading">
            Supported Disease Classes
          </h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm text-slate-300">
            <thead className="bg-slate-950 text-slate-400 uppercase text-[11px] font-semibold border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Disease Class</th>
                <th className="py-3 px-4">Pathogen / Scientific Name</th>
                <th className="py-3 px-4">Category</th>
                <th className="py-3 px-4">Risk Level</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80 font-sans">
              {MANGO_DISEASES.map((d) => (
                <tr key={d.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3.5 px-4 font-bold text-white flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                    {d.name}
                  </td>
                  <td className="py-3.5 px-4 font-mono italic text-slate-400 text-xs">
                    {d.scientificName}
                  </td>
                  <td className="py-3.5 px-4">
                    <span className="px-2 py-0.5 rounded text-xs bg-slate-800 border border-slate-700 text-slate-300">
                      {d.category}
                    </span>
                  </td>
                  <td className="py-3.5 px-4">
                    <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                      d.risk === 'None' ? 'text-emerald-400' :
                      d.risk === 'Low' ? 'text-blue-400' :
                      d.risk === 'Moderate' ? 'text-purple-400' : 'text-rose-400'
                    }`}>
                      {d.risk}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 4: Future Scope */}
      <div className="rounded-3xl bg-slate-900/80 border border-slate-800 p-8 md:p-10 space-y-8 shadow-xl">
        <div className="flex items-center gap-3 pb-4 border-b border-slate-800">
          <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white font-heading">
              Future Scope & Roadmap
            </h2>
            <p className="text-xs text-slate-400">Planned Extensions for Real-World Deployment</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {futureScopeItems.map((item, index) => {
            const Icon = item.icon;
            return (
              <div
                key={index}
                className="p-6 rounded-2xl bg-slate-950/60 border border-slate-800/80 hover:border-emerald-500/40 space-y-3 transition-colors"
              >
                <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                  <Icon className="w-5 h-5 text-emerald-400" />
                </div>
                <h3 className="text-base font-bold text-white font-heading">
                  {item.title}
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {item.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* CTA Bottom Banner */}
      <div className="text-center p-8 rounded-3xl bg-gradient-to-r from-emerald-950/40 via-slate-900 to-teal-950/40 border border-emerald-500/30 space-y-4">
        <h3 className="text-2xl font-bold text-white font-heading">
          Experience Multi-Class Foliar Diagnosis
        </h3>
        <p className="text-slate-400 text-sm max-w-xl mx-auto">
          Test with preloaded disease samples or upload your own leaf specimen to inspect real-time multi-class Softmax breakdown.
        </p>
        <button
          onClick={onStartDetection}
          className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-sm shadow-lg shadow-emerald-500/20"
        >
          <Sparkles className="w-4 h-4 fill-slate-950" />
          <span>Launch Detection Page</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

    </div>
  );
}

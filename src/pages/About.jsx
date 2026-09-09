import React from 'react';
import { Cpu, Target, Layers, Rocket, ShieldCheck, Code2, CheckCircle2, ArrowRight, GitBranch } from 'lucide-react';
import { Button } from '../components/common/Button';

export const About = ({ onNavigate }) => {
  const futureScopeItems = [
    {
      title: 'Real-Time Camera Scanner',
      description: 'Continuous video feed leaf detection with on-device bounding boxes and instant condition tracking.',
      icon: Target
    },
    {
      title: 'Dedicated Mobile Application',
      description: 'Offline-capable progressive mobile app for field agronomists and farmers with geolocation tagging.',
      icon: Rocket
    },
    {
      title: 'Trained ML Model API Integration',
      description: 'Direct REST / gRPC pipeline connecting FastAPI, Flask, or TorchServe backend hosting the trained weights.',
      icon: Cpu
    },
    {
      title: 'Orchard Disease History & Telemetry',
      description: 'Historical disease trend tracking, geographic infection heatmaps, and seasonal outbreak warnings.',
      icon: Layers
    },
    {
      title: 'Ensemble Architecture & Higher Accuracy',
      description: 'Multi-scale Vision Transformers (ViT) and ConvNeXt architectures fine-tuned on diverse field datasets.',
      icon: ShieldCheck
    }
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-16">
      {/* Header Banner */}
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/70 border border-emerald-500/30 text-emerald-300 text-xs font-semibold shadow-md">
          <Code2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>Architecture & Future Roadmap</span>
        </div>
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-white tracking-tight">
          About MangoAI
        </h1>
        <p className="text-slate-300 text-base sm:text-lg leading-relaxed">
          Mango Leaf Disease Detection is an AI-based image classification system designed to identify different mango leaf disease categories from leaf images.
        </p>
      </div>

      {/* Grid: Project Objective & AI Classification Pipeline */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Project Objective */}
        <div className="rounded-3xl glass-panel p-8 border border-slate-800 space-y-4 shadow-xl">
          <div className="w-12 h-12 rounded-2xl bg-emerald-950/80 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Target className="w-6 h-6" />
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            Project Objective
          </h2>
          <p className="text-sm text-slate-300 leading-relaxed">
            The primary objective of this project is to bridge computer vision technology with precision agriculture. By providing immediate, accessible multi-class disease diagnostics on standard leaf photos, orchard growers and researchers can catch fungal and bacterial outbreaks early, minimizing crop yield loss and optimizing targeted fungicide applications.
          </p>
          <div className="space-y-2 pt-2 text-xs text-slate-400">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Multi-class classification covering 6 key health states</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Transparent confidence distribution across all potential pathologies</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Actionable, non-prescriptive cultural agronomy next steps</span>
            </div>
          </div>
        </div>

        {/* How AI Classification Works */}
        <div className="rounded-3xl glass-panel p-8 border border-slate-800 space-y-4 shadow-xl">
          <div className="w-12 h-12 rounded-2xl bg-teal-950/80 border border-teal-500/30 flex items-center justify-center text-teal-400">
            <Cpu className="w-6 h-6" />
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            How AI Classification Works
          </h2>
          <p className="text-sm text-slate-300 leading-relaxed">
            The detection pipeline utilizes Deep Convolutional Neural Networks (CNNs) trained on thousands of verified botanical mango leaf images:
          </p>
          <ol className="space-y-3 pt-1 text-xs text-slate-300">
            <li className="flex items-start gap-2.5">
              <span className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-[11px] flex-shrink-0 mt-0.5">1</span>
              <div>
                <strong className="text-white">Preprocessing & Normalization:</strong> Input images are resized to 224×224 pixels and standardized across RGB color channels.
              </div>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-[11px] flex-shrink-0 mt-0.5">2</span>
              <div>
                <strong className="text-white">Convolutional Feature Extraction:</strong> Deep layers capture low-level edges, chlorotic rings, angular margins, and necrotic lesion textures.
              </div>
            </li>
            <li className="flex items-start gap-2.5">
              <span className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-[11px] flex-shrink-0 mt-0.5">3</span>
              <div>
                <strong className="text-white">Softmax Classification:</strong> Fully connected layers compute normalized probability distribution scores across all target disease classes.
              </div>
            </li>
          </ol>
        </div>
      </div>

      {/* ML Integration Specification & Code Schema Block */}
      <div className="rounded-3xl glass-panel p-8 border border-slate-800 space-y-6 shadow-2xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-emerald-400">
              <GitBranch className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">
                Future ML API Integration Architecture
              </h3>
              <p className="text-xs text-slate-400">
                Plug-and-play contract ready for trained model deployment
              </p>
            </div>
          </div>
          <span className="text-xs font-mono px-3 py-1 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-500/30">
            POST /api/v1/predict
          </span>
        </div>

        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
          The frontend components communicate with an abstracted service layer (<code className="text-emerald-400">src/services/predictionService.js</code>). Once your trained ML backend is running, simply update the prediction function to return the standard multi-class JSON payload below:
        </p>

        {/* JSON Schema Display */}
        <div className="rounded-2xl bg-slate-950 p-4 sm:p-5 border border-slate-800 overflow-x-auto text-xs font-mono text-emerald-300">
          <pre>{`// Expected Model Response JSON Schema:
{
  "disease": "Anthracnose",
  "confidence": 94.6,
  "status": "Disease Detected",
  "risk": "Moderate",
  "executionTimeMs": 128,
  "predictions": [
    { "name": "Anthracnose", "confidence": 94.6 },
    { "name": "Bacterial Canker", "confidence": 2.8 },
    { "name": "Powdery Mildew", "confidence": 1.4 },
    { "name": "Healthy", "confidence": 0.8 },
    { "name": "Die Back", "confidence": 0.3 },
    { "name": "Sooty Mold", "confidence": 0.1 }
  ]
}`}</pre>
        </div>
      </div>

      {/* Future Scope Section */}
      <div className="space-y-8">
        <div className="text-center max-w-2xl mx-auto space-y-2">
          <h3 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Future Scope & Roadmap
          </h3>
          <p className="text-slate-400 text-sm">
            Upcoming expansions for enterprise agricultural deployment.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {futureScopeItems.map((item, i) => {
            const Icon = item.icon;
            return (
              <div
                key={i}
                className="rounded-2xl glass-panel p-6 border border-slate-800 space-y-3 group hover:border-emerald-500/40 transition-colors"
              >
                <div className="w-10 h-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-emerald-400 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
                  <Icon className="w-5 h-5" />
                </div>
                <h4 className="text-base font-bold text-white group-hover:text-emerald-300 transition-colors">
                  {item.title}
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {item.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Bottom CTA */}
      <div className="text-center pt-6">
        <Button
          variant="primary"
          size="lg"
          onClick={() => onNavigate('detection')}
          iconRight={ArrowRight}
        >
          Try the Disease Detector
        </Button>
      </div>
    </div>
  );
};

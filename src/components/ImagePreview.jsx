import React from 'react';
import { Trash2, RefreshCw, Sparkles, FileText, HardDrive, CheckCircle2, ArrowRight } from 'lucide-react';

export default function ImagePreview({ imageData, onRemove, onReplace, onAnalyze, isAnalyzing }) {
  if (!imageData) return null;

  return (
    <div className="w-full rounded-3xl bg-slate-900/80 border border-slate-800 p-6 md:p-8 space-y-6 shadow-2xl animate-fadeIn">
      
      {/* Top Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
          <span className="text-sm font-bold text-white tracking-wide">Image Prepared for Analysis</span>
        </div>

        <button
          onClick={onRemove}
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-rose-400 hover:text-rose-300 transition-colors"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Remove image</span>
        </button>
      </div>

      {/* Main Preview & Meta Grid */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-center">
        
        {/* Large Image Frame */}
        <div className="md:col-span-6 relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 aspect-square max-h-[360px] flex items-center justify-center group shadow-inner">
          <img
            src={imageData.previewUrl}
            alt="Prepared Mango Leaf"
            className="w-full h-full object-contain p-2 group-hover:scale-105 transition-transform duration-500 select-none"
          />

          {/* AI Focus Corner Indicators */}
          <div className="absolute top-3 left-3 w-4 h-4 border-t-2 border-l-2 border-emerald-400/80"></div>
          <div className="absolute top-3 right-3 w-4 h-4 border-t-2 border-r-2 border-emerald-400/80"></div>
          <div className="absolute bottom-3 left-3 w-4 h-4 border-b-2 border-l-2 border-emerald-400/80"></div>
          <div className="absolute bottom-3 right-3 w-4 h-4 border-b-2 border-r-2 border-emerald-400/80"></div>
        </div>

        {/* Metadata Details & Action Buttons */}
        <div className="md:col-span-6 flex flex-col justify-between space-y-6">
          
          <div className="space-y-4">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              FILE SPECIFICATIONS
            </h4>

            {/* Metadata Rows */}
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 text-xs">
                <span className="text-slate-400 font-medium">File Name:</span>
                <span className="font-mono text-slate-200 font-semibold truncate max-w-[200px]">
                  {imageData.fileName || 'specimen_input.png'}
                </span>
              </div>

              <div className="flex items-center justify-between p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 text-xs">
                <span className="text-slate-400 font-medium">File Size:</span>
                <span className="font-mono text-slate-200 font-semibold">
                  {imageData.fileSize || '2.23 MB'}
                </span>
              </div>

              <div className="flex items-center justify-between p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 text-xs">
                <span className="text-slate-400 font-medium">MIME Type:</span>
                <span className="font-mono text-slate-200 font-semibold uppercase">
                  {imageData.fileType ? imageData.fileType.toUpperCase() : 'IMAGE/PNG'}
                </span>
              </div>
            </div>
          </div>

          {/* Action CTAs */}
          <div className="space-y-3 pt-2">
            <button
              onClick={onAnalyze}
              disabled={isAnalyzing}
              className="w-full flex items-center justify-center gap-2.5 py-4 px-6 rounded-2xl bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-slate-950 font-bold text-base shadow-xl shadow-emerald-500/25 hover:scale-[1.01] active:scale-100 transition-all duration-200"
            >
              <Sparkles className="w-5 h-5 text-slate-950 fill-slate-950" />
              <span>Analyze Leaf</span>
            </button>

            <button
              onClick={onReplace}
              className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-slate-950 hover:bg-slate-850 text-slate-300 hover:text-white font-semibold text-xs border border-slate-800 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5 text-slate-400" />
              <span>Choose Different Image</span>
            </button>
          </div>

        </div>

      </div>

    </div>
  );
}

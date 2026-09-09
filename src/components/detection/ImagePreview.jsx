import React from 'react';
import { Trash2, RefreshCw, Sparkles, FileText, CheckCircle, ShieldCheck } from 'lucide-react';
import { Button } from '../common/Button';

export const ImagePreview = ({ image, onRemove, onAnalyze, isAnalyzing }) => {
  if (!image) return null;

  return (
    <div className="rounded-3xl glass-panel p-6 sm:p-8 border border-slate-700/80 shadow-2xl space-y-6">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
          <h3 className="text-base font-bold text-white">Image Prepared for Analysis</h3>
        </div>
        <button
          onClick={onRemove}
          className="inline-flex items-center gap-1.5 text-xs text-rose-400 hover:text-rose-300 transition-colors p-1.5 rounded-lg hover:bg-rose-950/40"
          title="Remove selected image"
        >
          <Trash2 className="w-4 h-4" />
          <span className="hidden sm:inline">Remove image</span>
        </button>
      </div>

      {/* Main Image View & Metadata Grid */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
        {/* Large Image Preview */}
        <div className="md:col-span-7 relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 shadow-inner group aspect-video sm:aspect-4/3 flex items-center justify-center">
          <img
            src={image.previewUrl}
            alt="Selected Mango Leaf"
            className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-500"
          />
          {image.isSample && (
            <div className="absolute top-3 left-3 px-2.5 py-1 rounded-full bg-slate-950/80 border border-emerald-500/40 text-[11px] font-medium text-emerald-300 backdrop-blur-md">
              Sample Specimen: {image.sampleInfo?.categoryLabel}
            </div>
          )}
        </div>

        {/* Metadata & Actions Column */}
        <div className="md:col-span-5 flex flex-col justify-between h-full space-y-6">
          <div className="space-y-3 bg-slate-900/70 p-4 rounded-2xl border border-slate-800/80 text-xs">
            <h4 className="font-bold text-slate-200 uppercase tracking-wider text-[11px]">
              File Specifications
            </h4>
            <div className="space-y-2 text-slate-300">
              <div className="flex items-center justify-between py-1 border-b border-slate-800">
                <span className="text-slate-500">File Name:</span>
                <span className="font-mono font-medium truncate max-w-[170px] text-right" title={image.name}>
                  {image.name}
                </span>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-slate-800">
                <span className="text-slate-500">File Size:</span>
                <span className="font-mono">{image.size}</span>
              </div>
              <div className="flex items-center justify-between py-1">
                <span className="text-slate-500">MIME Type:</span>
                <span className="font-mono uppercase text-emerald-400">{image.type}</span>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <Button
              variant="primary"
              size="lg"
              icon={Sparkles}
              onClick={onAnalyze}
              disabled={isAnalyzing}
              className="w-full justify-center shadow-xl shadow-emerald-950/80 text-base"
            >
              Analyze Leaf
            </Button>

            <Button
              variant="outline"
              size="md"
              icon={RefreshCw}
              onClick={onRemove}
              disabled={isAnalyzing}
              className="w-full justify-center text-xs"
            >
              Choose Different Image
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

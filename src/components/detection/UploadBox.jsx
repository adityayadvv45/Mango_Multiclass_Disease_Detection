import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Sparkles, Check, AlertCircle } from 'lucide-react';
import { SAMPLE_LEAF_IMAGES } from '../../data/sampleImages';

export const UploadBox = ({ onImageSelected, onError }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const validateAndProcessFile = (file) => {
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      if (onError) {
        onError('Please upload a JPG, JPEG, or PNG image.');
      }
      return;
    }

    // 10MB limit check
    if (file.size > 10 * 1024 * 1024) {
      if (onError) {
        onError('Image size exceeds 10MB limit. Please upload a smaller photo.');
      }
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      onImageSelected({
        file,
        previewUrl: e.target.result,
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
        type: file.type || 'image/jpeg',
        isSample: false
      });
    };
    reader.onerror = () => {
      if (onError) {
        onError('Failed to read selected image file. Please try another.');
      }
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const handleSelectSample = (sample) => {
    onImageSelected({
      file: null,
      previewUrl: sample.dataUrl,
      name: sample.fileName,
      size: sample.fileSize,
      type: sample.fileType,
      isSample: true,
      diseaseId: sample.diseaseId,
      sampleInfo: sample
    });
  };

  return (
    <div className="space-y-8">
      {/* Upload Drop Zone Card */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative cursor-pointer rounded-3xl border-2 border-dashed transition-all duration-300 p-8 sm:p-14 text-center group ${
          isDragging
            ? 'border-emerald-400 bg-emerald-950/40 scale-[1.01] shadow-2xl shadow-emerald-950/80'
            : 'border-slate-700/80 hover:border-emerald-500/60 bg-slate-900/50 hover:bg-slate-900/80 shadow-xl'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp"
          onChange={handleFileChange}
          className="hidden"
          id="mango-leaf-input"
        />

        {/* Upload Center Visual */}
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="relative">
            <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-3xl bg-slate-800/90 border border-slate-700 flex items-center justify-center text-emerald-400 group-hover:bg-emerald-600 group-hover:text-white group-hover:scale-110 group-hover:rotate-3 transition-all duration-300 shadow-xl shadow-slate-950/80">
              <UploadCloud className="w-10 h-10 sm:w-12 sm:h-12" />
            </div>
            <div className="absolute -bottom-1 -right-1 w-7 h-7 rounded-xl bg-emerald-500 text-slate-950 flex items-center justify-center shadow-lg font-bold">
              <Sparkles className="w-4 h-4" />
            </div>
          </div>

          <div className="space-y-1.5 max-w-md">
            <h3 className="text-xl sm:text-2xl font-bold text-white group-hover:text-emerald-300 transition-colors">
              Drop your mango leaf image here
            </h3>
            <p className="text-sm font-medium text-slate-400">
              or <span className="text-emerald-400 underline underline-offset-4 group-hover:text-emerald-300">click to browse</span> from your device
            </p>
          </div>

          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/80 border border-slate-700/60 text-slate-400 text-xs">
            <span>Supports JPG, JPEG, PNG, WEBP</span>
            <span>•</span>
            <span>Max 10MB</span>
          </div>

          <p className="text-xs text-slate-500 italic max-w-sm pt-2">
            "For best results, use a clear image where the leaf is visible."
          </p>
        </div>
      </div>

      {/* Sample Specimens Tray for Fast 1-Click Evaluation */}
      <div className="rounded-2xl glass-panel p-6 border border-slate-800">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <ImageIcon className="w-4 h-4 text-emerald-400" />
            <h4 className="text-sm font-bold text-slate-200">
              Quick Test: Select a Sample Mango Leaf Specimen
            </h4>
          </div>
          <span className="text-xs text-slate-400 hidden sm:inline">
            1-Click Demo Testing
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {SAMPLE_LEAF_IMAGES.map((sample) => (
            <button
              key={sample.id}
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                handleSelectSample(sample);
              }}
              className="flex flex-col items-center p-3 rounded-xl bg-slate-900/80 hover:bg-emerald-950/40 border border-slate-800 hover:border-emerald-500/40 transition-all duration-200 group text-left hover:scale-[1.02]"
            >
              <div className="w-full aspect-video rounded-lg overflow-hidden bg-slate-950 mb-2.5 border border-slate-800 relative">
                <img
                  src={sample.thumbnail}
                  alt={sample.name}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <span className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-slate-950/80 text-[9px] font-mono text-slate-300">
                  {sample.categoryLabel}
                </span>
              </div>
              <span className="text-xs font-semibold text-slate-200 group-hover:text-emerald-400 transition-colors w-full truncate">
                {sample.name.replace(' Sample Leaf', '')}
              </span>
              <span className="text-[10px] text-slate-400 w-full truncate">
                {sample.fileSize}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

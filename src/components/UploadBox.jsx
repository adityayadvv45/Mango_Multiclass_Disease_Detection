import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, AlertCircle, Sparkles, CheckCircle2, Zap } from 'lucide-react';
import { SAMPLE_LEAF_IMAGES } from '../data/sampleImages';

export default function UploadBox({ onImageSelected, onSampleSelected }) {
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const fileInputRef = useRef(null);

  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  const maxSizeBytes = 10 * 1024 * 1024; // 10MB

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
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file) => {
    setErrorMessage('');

    // Format validation
    if (!allowedTypes.includes(file.type.toLowerCase())) {
      setErrorMessage('Please upload a JPG, JPEG, PNG, or WEBP image of a mango leaf.');
      return;
    }

    // Size validation
    if (file.size > maxSizeBytes) {
      setErrorMessage('File size exceeds 10MB. Please select a smaller leaf image.');
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const fileData = {
        file: file,
        fileName: file.name,
        fileSize: formatFileSize(file.size),
        fileType: file.type,
        previewUrl: reader.result,
        isCustomUpload: true
      };
      onImageSelected(fileData);
    };
    reader.onerror = () => {
      setErrorMessage('Failed to read image file. Please try selecting another file.');
    };
    reader.readAsDataURL(file);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="w-full space-y-8 animate-fadeIn">
      
      {/* Drop Zone Box */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
        className={`relative group cursor-pointer rounded-3xl border-2 border-dashed p-8 md:p-12 text-center transition-all duration-300 ${
          isDragging
            ? 'border-emerald-400 bg-emerald-950/40 scale-[1.01] shadow-2xl shadow-emerald-500/20'
            : 'border-slate-700/80 bg-slate-900/50 hover:border-emerald-500/50 hover:bg-slate-900/80'
        }`}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileInputChange}
          accept=".jpg,.jpeg,.png,.webp"
          className="hidden"
          aria-label="Upload Mango Leaf Image"
        />

        <div className="max-w-md mx-auto flex flex-col items-center justify-center space-y-4">
          
          {/* Icon Badge */}
          <div className={`w-20 h-20 rounded-2xl flex items-center justify-center transition-all duration-300 ${
            isDragging 
              ? 'bg-emerald-500 text-slate-950 scale-110 shadow-lg shadow-emerald-500/40' 
              : 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 group-hover:scale-105 group-hover:bg-emerald-500/20'
          }`}>
            <UploadCloud className="w-10 h-10 transform group-hover:-translate-y-1 transition-transform" />
          </div>

          {/* Text Instructions */}
          <div className="space-y-1.5">
            <h3 className="text-xl md:text-2xl font-bold text-white font-heading">
              Drop your mango leaf image here
            </h3>
            <p className="text-sm md:text-base text-slate-400">
              or <span className="text-emerald-400 font-semibold underline underline-offset-4 group-hover:text-emerald-300">click to browse</span> from your device
            </p>
          </div>

          {/* Formats and Note */}
          <div className="pt-2 flex flex-col items-center gap-2">
            <div className="flex items-center gap-2 text-xs font-medium text-slate-400 bg-slate-800/80 px-3.5 py-1.5 rounded-full border border-slate-700/60">
              <ImageIcon className="w-3.5 h-3.5 text-emerald-400" />
              <span>Supports JPG, JPEG, PNG, WEBP • Max 10MB</span>
            </div>
            <p className="text-xs text-slate-400 max-w-sm text-center">
              "For best results, use a clear image where the leaf is visible."
            </p>
          </div>

        </div>

        {/* Ambient Hover Ring */}
        <div className="absolute inset-0 rounded-3xl pointer-events-none border border-emerald-500/0 group-hover:border-emerald-500/20 transition-colors"></div>
      </div>

      {/* Inline Error Display */}
      {errorMessage && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm animate-fadeIn">
          <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400" />
          <p>{errorMessage}</p>
        </div>
      )}

      {/* Quick Test Preset Gallery */}
      <div className="pt-4 border-t border-slate-800/80 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm font-bold text-slate-200">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <span>Quick Test: Select a Sample Mango Leaf Specimen</span>
          </div>
          <span className="text-xs text-slate-400 hidden sm:inline-block">1-Click Demo Testing</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-4 gap-3.5">
          {SAMPLE_LEAF_IMAGES.map((sample) => (
            <button
              key={sample.id}
              onClick={() => onSampleSelected(sample)}
              className={`group/sample flex flex-col items-center text-center p-3 rounded-2xl border transition-all duration-200 ${
                sample.isMultiPathology
                  ? 'bg-gradient-to-b from-amber-950/30 to-slate-900 border-amber-500/40 hover:border-amber-400 hover:scale-[1.02] shadow-lg shadow-amber-950/20'
                  : 'bg-slate-900/60 border-slate-800 hover:border-emerald-500/50 hover:bg-slate-850 hover:scale-[1.02]'
              }`}
            >
              <div className="w-full aspect-square rounded-xl overflow-hidden bg-slate-950 mb-2.5 border border-slate-800 relative">
                <img 
                  src={sample.previewUrl} 
                  alt={sample.name} 
                  className="w-full h-full object-cover group-hover/sample:scale-105 transition-transform duration-300" 
                />
                {sample.isMultiPathology && (
                  <div className="absolute top-1.5 right-1.5 px-2 py-0.5 rounded-md bg-amber-500 text-slate-950 font-black text-[9px] uppercase tracking-tight shadow-md">
                    Featured Demo
                  </div>
                )}
              </div>
              <span className="text-xs font-bold text-slate-200 group-hover/sample:text-emerald-300 truncate w-full">
                {sample.name}
              </span>
              <span className="text-[10px] text-slate-400 mt-0.5">
                {sample.expectedDisease}
              </span>
            </button>
          ))}
        </div>
      </div>

    </div>
  );
}

import React from 'react';
import { AlertCircle, CheckCircle, Info, X } from 'lucide-react';

export const Toast = ({ message, type = 'error', onClose }) => {
  if (!message) return null;

  const icons = {
    error: AlertCircle,
    success: CheckCircle,
    info: Info
  };

  const styleMap = {
    error: 'bg-rose-950/90 text-rose-200 border-rose-500/50 shadow-rose-950/50',
    success: 'bg-emerald-950/90 text-emerald-200 border-emerald-500/50 shadow-emerald-950/50',
    info: 'bg-slate-900/90 text-slate-200 border-cyan-500/50 shadow-cyan-950/50'
  };

  const IconComponent = icons[type] || icons.error;

  return (
    <div className={`fixed bottom-6 right-6 z-50 flex items-center gap-3 px-4 py-3 rounded-xl border backdrop-blur-md shadow-xl transition-all duration-300 animate-in fade-in slide-in-from-bottom-5 ${styleMap[type]}`}>
      <IconComponent className="w-5 h-5 flex-shrink-0" />
      <span className="text-sm font-medium">{message}</span>
      {onClose && (
        <button
          onClick={onClose}
          className="ml-2 text-current opacity-70 hover:opacity-100 p-1 rounded-lg hover:bg-white/10 transition-colors"
          aria-label="Close notification"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};

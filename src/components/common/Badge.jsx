import React from 'react';

export const Badge = ({ children, variant = 'default', size = 'md', className = '' }) => {
  const variantStyles = {
    default: 'bg-slate-800/80 text-slate-300 border-slate-700/60',
    success: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    warning: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    danger: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
    info: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30',
    purple: 'bg-purple-500/15 text-purple-300 border-purple-500/30',
    brand: 'bg-brand-500/15 text-brand-300 border-brand-500/30'
  };

  const sizeStyles = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-xs font-medium',
    lg: 'px-3 py-1.5 text-sm font-medium'
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border backdrop-blur-sm transition-colors ${
        variantStyles[variant] || variantStyles.default
      } ${sizeStyles[size] || sizeStyles.md} ${className}`}
    >
      {children}
    </span>
  );
};

export const RiskBadge = ({ risk = 'Low', className = '' }) => {
  const normalized = risk.toLowerCase();
  let variant = 'default';
  let dotColor = 'bg-slate-400';

  if (normalized === 'none' || normalized === 'healthy') {
    variant = 'success';
    dotColor = 'bg-emerald-400';
  } else if (normalized === 'low') {
    variant = 'info';
    dotColor = 'bg-cyan-400';
  } else if (normalized === 'moderate' || normalized.includes('moderate')) {
    variant = 'warning';
    dotColor = 'bg-amber-400';
  } else if (normalized === 'high') {
    variant = 'danger';
    dotColor = 'bg-rose-400';
  }

  return (
    <Badge variant={variant} className={className}>
      <span className={`w-1.5 h-1.5 rounded-full ${dotColor} animate-pulse`} />
      <span>{risk} Risk</span>
    </Badge>
  );
};

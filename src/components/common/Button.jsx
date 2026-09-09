import React from 'react';

export const Button = ({
  children,
  variant = 'primary',
  size = 'md',
  onClick,
  disabled = false,
  type = 'button',
  icon: Icon,
  iconRight: IconRight,
  className = '',
  ...props
}) => {
  const baseStyles = 'inline-flex items-center justify-center font-medium transition-all duration-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed select-none active:scale-[0.98]';

  const variantStyles = {
    primary: 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-900/40 hover:shadow-emerald-900/60 focus:ring-emerald-500 border border-emerald-500/30',
    secondary: 'bg-slate-800/90 hover:bg-slate-700/90 text-slate-200 hover:text-white border border-slate-700/80 focus:ring-slate-600 shadow-sm',
    outline: 'bg-transparent hover:bg-slate-800/60 text-slate-300 hover:text-white border border-slate-700 hover:border-slate-500 focus:ring-slate-500',
    ghost: 'bg-transparent hover:bg-slate-800/50 text-slate-400 hover:text-slate-100 focus:ring-slate-500',
    danger: 'bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-500/30 focus:ring-rose-500'
  };

  const sizeStyles = {
    sm: 'px-3 py-1.5 text-xs gap-1.5',
    md: 'px-4 py-2.5 text-sm gap-2',
    lg: 'px-6 py-3.5 text-base gap-2.5 font-semibold'
  };

  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      className={`${baseStyles} ${variantStyles[variant] || variantStyles.primary} ${sizeStyles[size] || sizeStyles.md} ${className}`}
      {...props}
    >
      {Icon && <Icon className={size === 'sm' ? 'w-3.5 h-3.5' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} />}
      {children}
      {IconRight && <IconRight className={size === 'sm' ? 'w-3.5 h-3.5' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} />}
    </button>
  );
};

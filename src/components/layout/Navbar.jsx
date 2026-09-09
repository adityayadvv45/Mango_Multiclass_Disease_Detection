import React, { useState } from 'react';
import { Leaf, Menu, X, Sparkles, Activity, BookOpen, Info, ShieldCheck } from 'lucide-react';
import { Button } from '../common/Button';

export const Navbar = ({ activePage, onNavigate }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'home', label: 'Home', icon: Leaf },
    { id: 'detection', label: 'Detection', icon: Activity },
    { id: 'diseases', label: 'Diseases', icon: BookOpen },
    { id: 'about', label: 'About', icon: Info },
  ];

  const handleNavClick = (id) => {
    onNavigate(id);
    setMobileMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 sm:h-20">
          {/* Brand Logo */}
          <div
            onClick={() => handleNavClick('home')}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-700 text-white shadow-lg shadow-emerald-950/50 group-hover:scale-105 transition-transform">
              <Leaf className="w-5 h-5 text-white" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full border-2 border-slate-950 animate-ping" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full border-2 border-slate-950" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-extrabold tracking-tight bg-gradient-to-r from-emerald-400 via-teal-300 to-emerald-200 bg-clip-text text-transparent">
                  MangoAI
                </span>
                <span className="px-1.5 py-0.5 text-[10px] uppercase font-bold tracking-wider rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  ML Vision
                </span>
              </div>
              <p className="text-[11px] text-slate-400 -mt-0.5 hidden sm:block">
                Multi-Class Leaf Disease Classification
              </p>
            </div>
          </div>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center gap-1 bg-slate-900/60 p-1.5 rounded-2xl border border-slate-800/60 shadow-inner">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-emerald-600/90 text-white shadow-md shadow-emerald-950/40 border border-emerald-500/40'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-100' : 'text-slate-400'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Desktop CTA Action */}
          <div className="hidden md:flex items-center gap-3">
            <Button
              variant="primary"
              size="md"
              icon={Sparkles}
              onClick={() => handleNavClick('detection')}
            >
              Detect Disease
            </Button>
          </div>

          {/* Mobile Menu Button */}
          <div className="flex md:hidden items-center gap-2">
            <Button
              variant="primary"
              size="sm"
              icon={Sparkles}
              onClick={() => handleNavClick('detection')}
            >
              Detect
            </Button>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              aria-label="Toggle Navigation Menu"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden border-b border-slate-800 bg-slate-950/95 backdrop-blur-2xl px-4 pt-2 pb-6 space-y-2 animate-in slide-in-from-top-4">
          <div className="grid grid-cols-2 gap-2 pt-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`flex items-center gap-2.5 px-4 py-3 rounded-xl text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-emerald-600 text-white font-semibold'
                      : 'bg-slate-900/60 text-slate-300 hover:bg-slate-800 hover:text-white border border-slate-800'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </button>
              );
            })}
          </div>
          <div className="pt-2">
            <Button
              variant="primary"
              size="md"
              icon={Sparkles}
              onClick={() => handleNavClick('detection')}
              className="w-full justify-center"
            >
              Start Disease Detection
            </Button>
          </div>
        </div>
      )}
    </header>
  );
};

import React from 'react';
import { Search, Filter, X } from 'lucide-react';
import { CATEGORIES } from '../../data/diseases';

export const DiseaseFilter = ({
  searchQuery,
  onSearchChange,
  selectedCategory,
  onSelectCategory
}) => {
  return (
    <div className="flex flex-col md:flex-row items-center justify-between gap-4 bg-slate-900/60 p-3 sm:p-4 rounded-2xl border border-slate-800 backdrop-blur-md">
      {/* Category Filter Pills */}
      <div className="flex items-center gap-1.5 overflow-x-auto w-full md:w-auto pb-2 md:pb-0 scrollbar-none">
        {CATEGORIES.map((cat) => {
          const isSelected = selectedCategory === cat.id;
          return (
            <button
              key={cat.id}
              onClick={() => onSelectCategory(cat.id)}
              className={`px-3.5 py-1.5 rounded-xl text-xs font-medium whitespace-nowrap transition-all duration-150 ${
                isSelected
                  ? 'bg-emerald-600 text-white shadow-md shadow-emerald-950/60 border border-emerald-500/40'
                  : 'bg-slate-800/60 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-transparent'
              }`}
            >
              {cat.label}
            </button>
          );
        })}
      </div>

      {/* Search Input */}
      <div className="relative w-full md:w-72">
        <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search diseases or symptoms..."
          className="w-full pl-9 pr-8 py-2 bg-slate-950/80 rounded-xl text-xs text-slate-200 placeholder-slate-500 border border-slate-800 focus:border-emerald-500/50 focus:outline-none focus:ring-1 focus:ring-emerald-500/50 transition-all"
        />
        {searchQuery && (
          <button
            onClick={() => onSearchChange('')}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 p-0.5 rounded"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
};

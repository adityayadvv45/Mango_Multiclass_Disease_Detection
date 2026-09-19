import React, { useState } from 'react';
import { MANGO_DISEASES, CATEGORIES } from '../data/diseases';
import DiseaseCard from '../components/DiseaseCard';
import { SAMPLE_LEAF_IMAGES } from '../data/sampleImages';
import { BookOpen, Search, Filter, Sparkles, ShieldCheck } from 'lucide-react';

export default function Diseases({ onSelectDiseaseForTest }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');

  // Filter diseases based on search query and category
  const filteredDiseases = MANGO_DISEASES.filter((disease) => {
    const matchesSearch = 
      disease.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      disease.scientificName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      disease.shortDescription.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesCategory = 
      selectedCategory === 'All' || 
      disease.category.toLowerCase().includes(selectedCategory.toLowerCase());

    return matchesSearch && matchesCategory;
  });

  const handleTestDisease = (disease) => {
    const sample = SAMPLE_LEAF_IMAGES.find(
      s => s.expectedDisease.toLowerCase() === disease.name.toLowerCase()
    ) || SAMPLE_LEAF_IMAGES[0];

    if (onSelectDiseaseForTest) {
      onSelectDiseaseForTest(sample);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 md:py-16 space-y-12">
      
      {/* Page Header */}
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
          <BookOpen className="w-3.5 h-3.5" />
          <span>Pathology Reference Guide</span>
        </div>
        <h1 className="text-3xl sm:text-4xl md:text-5xl font-black text-white font-heading tracking-tight">
          Mango Leaf Diseases Catalog
        </h1>
        <p className="text-slate-400 text-base sm:text-lg">
          Explore the symptoms, pathological characteristics, and visual indicators of supported mango leaf conditions.
        </p>
      </div>

      {/* Filter and Search Bar Controls */}
      <div className="flex flex-col md:flex-row items-center justify-between gap-4 p-4 rounded-2xl bg-slate-900/80 border border-slate-800">
        
        {/* Search Input */}
        <div className="relative w-full md:w-80">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by disease or pathogen..."
            className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500/60 transition-colors"
          />
        </div>

        {/* Category Pills */}
        <div className="flex flex-wrap items-center gap-2 w-full md:w-auto">
          <span className="text-xs font-semibold text-slate-400 mr-1 hidden lg:inline">Category:</span>
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
                selectedCategory === cat
                  ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20'
                  : 'bg-slate-950 text-slate-300 border border-slate-800 hover:bg-slate-800'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

      </div>

      {/* Disease Cards Grid */}
      {filteredDiseases.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredDiseases.map((disease) => (
            <DiseaseCard
              key={disease.id}
              disease={disease}
              onSelectForTest={handleTestDisease}
              compact={false}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-16 rounded-3xl bg-slate-900/40 border border-slate-800/80 space-y-3">
          <div className="w-12 h-12 rounded-2xl bg-slate-800 flex items-center justify-center mx-auto text-slate-400">
            <Search className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-white">No Matching Diseases Found</h3>
          <p className="text-sm text-slate-400">Try adjusting your search terms or category filters.</p>
          <button
            onClick={() => { setSearchQuery(''); setSelectedCategory('All'); }}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-emerald-400 border border-slate-700 mt-2"
          >
            Clear Filters
          </button>
        </div>
      )}

      {/* Educational Notice Banner */}
      <div className="p-6 rounded-2xl bg-emerald-950/20 border border-emerald-500/20 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <ShieldCheck className="w-6 h-6 text-emerald-400 flex-shrink-0" />
          <div>
            <h4 className="text-sm font-bold text-white">Configurable Diagnostic Taxonomy</h4>
            <p className="text-xs text-slate-400">
              New pathogens, regional strains, and physiological disorders can be registered via <code className="text-emerald-400">src/data/diseases.js</code>.
            </p>
          </div>
        </div>

        <button
          onClick={() => handleTestDisease(MANGO_DISEASES[1])}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-xs font-bold shadow-md shadow-emerald-500/20 flex-shrink-0"
        >
          <Sparkles className="w-3.5 h-3.5 fill-slate-950" />
          <span>Try Diagnostics Demo</span>
        </button>
      </div>

    </div>
  );
}

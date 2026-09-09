import React, { useState } from 'react';
import { DISEASE_CLASSES } from '../data/diseases';
import { DiseaseCard } from '../components/diseases/DiseaseCard';
import { DiseaseFilter } from '../components/diseases/DiseaseFilter';
import { BookOpen, Sparkles, AlertCircle } from 'lucide-react';
import { SAMPLE_LEAF_IMAGES } from '../data/sampleImages';

export const Diseases = ({ onSelectDiseaseToTest }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  // Filter diseases based on search query and category
  const filteredDiseases = DISEASE_CLASSES.filter((disease) => {
    // Category match
    const categoryMatch =
      selectedCategory === 'all' ||
      disease.categoryType?.toLowerCase() === selectedCategory.toLowerCase();

    // Search query match
    const query = searchQuery.toLowerCase().trim();
    if (!query) return categoryMatch;

    const nameMatch = disease.name.toLowerCase().includes(query);
    const sciNameMatch = disease.scientificName?.toLowerCase().includes(query);
    const descMatch = disease.shortDescription?.toLowerCase().includes(query);
    const indicatorMatch = disease.visualIndicators?.some((ind) =>
      ind.toLowerCase().includes(query)
    );

    return categoryMatch && (nameMatch || sciNameMatch || descMatch || indicatorMatch);
  });

  const handleTestSample = (diseaseId) => {
    if (onSelectDiseaseToTest) {
      onSelectDiseaseToTest(diseaseId);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-10">
      {/* Page Header */}
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/70 border border-emerald-500/30 text-emerald-300 text-xs font-semibold shadow-md">
          <BookOpen className="w-3.5 h-3.5 text-emerald-400" />
          <span>Botanical Pathology Reference</span>
        </div>
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-white tracking-tight">
          Mango Leaf Diseases Catalog
        </h1>
        <p className="text-slate-400 text-base sm:text-lg">
          Detailed diagnostic profiles, symptom patterns, and agronomic management guidelines for all 8 supported classification categories.
        </p>
      </div>

      {/* Filter and Search Bar */}
      <DiseaseFilter
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        selectedCategory={selectedCategory}
        onSelectCategory={setSelectedCategory}
      />

      {/* Disease Cards Grid */}
      {filteredDiseases.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
          {filteredDiseases.map((disease) => (
            <DiseaseCard
              key={disease.id}
              disease={disease}
              onTestSample={handleTestSample}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-16 bg-slate-900/40 rounded-3xl border border-slate-800 space-y-3">
          <AlertCircle className="w-10 h-10 text-slate-500 mx-auto" />
          <h3 className="text-lg font-bold text-white">No Diseases Found</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            No disease matched your search query "{searchQuery}". Try searching for terms like "fungal", "spots", "canker", or "powdery".
          </p>
        </div>
      )}
    </div>
  );
};

import React, { useState, useEffect } from 'react';
import { Navbar } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';
import { Home } from './pages/Home';
import { Detection } from './pages/Detection';
import { Diseases } from './pages/Diseases';
import { About } from './pages/About';

export function App() {
  const [activePage, setActivePage] = useState('home');
  const [selectedDiseaseSample, setSelectedDiseaseSample] = useState(null);

  // Scroll to top whenever page changes
  const handleNavigate = (pageId) => {
    setActivePage(pageId);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Allow test button in Diseases page to switch to Detection
  const handleSelectDiseaseToTest = (diseaseId) => {
    setSelectedDiseaseSample(diseaseId);
    setActivePage('detection');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-emerald-500 selection:text-white relative">
      {/* Background Decorative Ambient Gradients */}
      <div className="fixed top-0 left-1/4 w-[600px] h-[500px] bg-emerald-500/5 rounded-full blur-[120px] pointer-events-none -z-10" />
      <div className="fixed bottom-10 right-1/4 w-[500px] h-[400px] bg-teal-500/5 rounded-full blur-[100px] pointer-events-none -z-10" />

      {/* Sticky Global Navigation */}
      <Navbar
        activePage={activePage}
        onNavigate={handleNavigate}
      />

      {/* Main Content Area */}
      <main className="flex-1">
        {activePage === 'home' && (
          <Home onNavigate={handleNavigate} />
        )}

        {activePage === 'detection' && (
          <Detection
            initialSampleId={selectedDiseaseSample}
            onClearInitialSample={() => setSelectedDiseaseSample(null)}
          />
        )}

        {activePage === 'diseases' && (
          <Diseases onSelectDiseaseToTest={handleSelectDiseaseToTest} />
        )}

        {activePage === 'about' && (
          <About onNavigate={handleNavigate} />
        )}
      </main>

      {/* Global Footer */}
      <Footer onNavigate={handleNavigate} />
    </div>
  );
}

export default App;

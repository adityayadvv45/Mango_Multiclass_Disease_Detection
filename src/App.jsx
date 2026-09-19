import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Detection from './pages/Detection';
import Diseases from './pages/Diseases';
import About from './pages/About';

export default function App() {
  const [activePage, setActivePage] = useState('home');
  const [initialDetectionSample, setInitialDetectionSample] = useState(null);

  const handleNavigate = (pageId) => {
    setActivePage(pageId);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleTestSampleFromOtherPage = (sample) => {
    setInitialDetectionSample(sample);
    setActivePage('detection');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-sans selection:bg-emerald-500 selection:text-white">
      {/* Navigation Bar */}
      <Navbar activePage={activePage} setActivePage={handleNavigate} />

      {/* Main Content View Switcher */}
      <main className="flex-1 w-full">
        {activePage === 'home' && (
          <Home
            onNavigate={handleNavigate}
            onTestSample={handleTestSampleFromOtherPage}
          />
        )}

        {activePage === 'detection' && (
          <Detection
            key={initialDetectionSample ? initialDetectionSample.id : 'default-detection'}
            initialSample={initialDetectionSample}
          />
        )}

        {activePage === 'diseases' && (
          <Diseases
            onSelectDiseaseForTest={handleTestSampleFromOtherPage}
          />
        )}

        {activePage === 'about' && (
          <About
            onStartDetection={() => handleNavigate('detection')}
          />
        )}
      </main>

      {/* Global Footer */}
      <Footer setActivePage={handleNavigate} />
    </div>
  );
}

/**
 * Curated Sample Mango Leaf Images
 * Realistic stylized SVG data representations with realistic botanical markings.
 * Enables 1-click test flow for users and reviewers.
 */

// Helper to convert SVG string to data URL
const toDataUrl = (svgStr) => `data:image/svg+xml;utf8,${encodeURIComponent(svgStr)}`;

// 1. Healthy Leaf SVG
const healthySvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#061a12" />
    </linearGradient>
    <linearGradient id="leafGradH" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4ade80" />
      <stop offset="40%" stop-color="#16a34a" />
      <stop offset="100%" stop-color="#14532d" />
    </linearGradient>
    <filter id="shadow">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#000" flood-opacity="0.5"/>
    </filter>
  </defs>
  <rect width="600" height="450" fill="url(#bgGrad)"/>
  <path d="M 80 320 Q 180 250 240 220" stroke="#854d0e" stroke-width="8" stroke-linecap="round" fill="none"/>
  <g filter="url(#shadow)">
    <path d="M 120 280 C 180 120, 360 80, 520 130 C 440 280, 300 350, 120 280 Z" fill="url(#leafGradH)"/>
    <path d="M 120 280 Q 320 200 520 130" stroke="#bbf7d0" stroke-width="4" stroke-linecap="round" fill="none" opacity="0.6"/>
    <path d="M 220 245 Q 260 180 310 160" stroke="#bbf7d0" stroke-width="2" fill="none" opacity="0.4"/>
    <path d="M 270 230 Q 320 160 380 145" stroke="#bbf7d0" stroke-width="2" fill="none" opacity="0.4"/>
    <path d="M 330 210 Q 380 150 430 138" stroke="#bbf7d0" stroke-width="2" fill="none" opacity="0.4"/>
  </g>
</svg>`;

// 2. Anthracnose Leaf SVG
const anthracnoseSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1c1308" />
    </linearGradient>
    <linearGradient id="leafGradA" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22c55e" />
      <stop offset="40%" stop-color="#15803d" />
      <stop offset="100%" stop-color="#0f3d1e" />
    </linearGradient>
    <radialGradient id="lesionHalo1">
      <stop offset="0%" stop-color="#261b0c" />
      <stop offset="60%" stop-color="#451a03" />
      <stop offset="85%" stop-color="#ca8a04" />
      <stop offset="100%" stop-color="#ca8a04" stop-opacity="0" />
    </radialGradient>
    <filter id="shadowA">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#000" flood-opacity="0.5"/>
    </filter>
  </defs>
  <rect width="600" height="450" fill="url(#bgGrad2)"/>
  <path d="M 80 320 Q 180 250 240 220" stroke="#854d0e" stroke-width="8" stroke-linecap="round" fill="none"/>
  <g filter="url(#shadowA)">
    <path d="M 120 280 C 180 120, 360 80, 520 130 C 440 280, 300 350, 120 280 Z" fill="url(#leafGradA)"/>
    <path d="M 120 280 Q 320 200 520 130" stroke="#86efac" stroke-width="4" stroke-linecap="round" fill="none" opacity="0.5"/>
    <circle cx="280" cy="180" r="28" fill="url(#lesionHalo1)"/>
    <ellipse cx="280" cy="180" rx="16" ry="12" fill="#1c0f05" />
    <circle cx="360" cy="240" r="36" fill="url(#lesionHalo1)"/>
    <circle cx="360" cy="240" r="20" fill="#180b02"/>
    <circle cx="440" cy="170" r="22" fill="url(#lesionHalo1)"/>
    <circle cx="440" cy="170" r="12" fill="#1c0f05"/>
  </g>
</svg>`;

// 3. Bacterial Canker Leaf SVG
const bacterialCankerSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#240c12" />
    </linearGradient>
    <linearGradient id="leafGradBC" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#15803d" />
      <stop offset="50%" stop-color="#166534" />
      <stop offset="100%" stop-color="#064e3b" />
    </linearGradient>
    <filter id="shadowBC">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#000" flood-opacity="0.5"/>
    </filter>
  </defs>
  <rect width="600" height="450" fill="url(#bgGrad3)"/>
  <path d="M 80 320 Q 180 250 240 220" stroke="#854d0e" stroke-width="8" stroke-linecap="round" fill="none"/>
  <g filter="url(#shadowBC)">
    <path d="M 120 280 C 180 120, 360 80, 520 130 C 440 280, 300 350, 120 280 Z" fill="url(#leafGradBC)"/>
    <path d="M 120 280 Q 320 200 520 130" stroke="#86efac" stroke-width="4" stroke-linecap="round" fill="none" opacity="0.4"/>
    <polygon points="250,170 285,160 295,190 260,200" fill="#713f12" stroke="#eab308" stroke-width="2"/>
    <polygon points="340,210 375,195 385,230 350,240" fill="#713f12" stroke="#eab308" stroke-width="2"/>
    <polygon points="410,150 440,140 448,168 418,175" fill="#713f12" stroke="#eab308" stroke-width="2"/>
  </g>
</svg>`;

// 4. Powdery Mildew Leaf SVG
const powderyMildewSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad4" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e102a" />
    </linearGradient>
    <linearGradient id="leafGradPM" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22c55e" />
      <stop offset="50%" stop-color="#15803d" />
      <stop offset="100%" stop-color="#064e3b" />
    </linearGradient>
    <radialGradient id="powderyGrad1">
      <stop offset="0%" stop-color="#f8fafc" stop-opacity="0.85" />
      <stop offset="60%" stop-color="#cbd5e1" stop-opacity="0.4" />
      <stop offset="100%" stop-color="#94a3b8" stop-opacity="0" />
    </radialGradient>
    <filter id="shadowPM">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#000" flood-opacity="0.5"/>
    </filter>
  </defs>
  <rect width="600" height="450" fill="url(#bgGrad4)"/>
  <path d="M 80 320 Q 180 250 240 220" stroke="#854d0e" stroke-width="8" stroke-linecap="round" fill="none"/>
  <g filter="url(#shadowPM)">
    <path d="M 120 280 C 180 120, 360 80, 520 130 C 440 280, 300 350, 120 280 Z" fill="url(#leafGradPM)"/>
    <ellipse cx="260" cy="190" rx="45" ry="32" fill="url(#powderyGrad1)"/>
    <ellipse cx="370" cy="220" rx="55" ry="38" fill="url(#powderyGrad1)"/>
    <ellipse cx="440" cy="160" rx="38" ry="26" fill="url(#powderyGrad1)"/>
  </g>
</svg>`;

// 5. Multi-Disease Specimen SVG (Anthracnose Spots + Powdery Mildew Bloom)
const multiDiseaseSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGradM" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#2a1215" />
    </linearGradient>
    <linearGradient id="leafGradM" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22c55e" />
      <stop offset="50%" stop-color="#15803d" />
      <stop offset="100%" stop-color="#0f3d1e" />
    </linearGradient>
    <radialGradient id="lesionHaloM">
      <stop offset="0%" stop-color="#261b0c" />
      <stop offset="60%" stop-color="#451a03" />
      <stop offset="85%" stop-color="#ca8a04" />
      <stop offset="100%" stop-color="#ca8a04" stop-opacity="0" />
    </radialGradient>
    <radialGradient id="powderyGradM">
      <stop offset="0%" stop-color="#f8fafc" stop-opacity="0.85" />
      <stop offset="60%" stop-color="#cbd5e1" stop-opacity="0.35" />
      <stop offset="100%" stop-color="#94a3b8" stop-opacity="0" />
    </radialGradient>
    <filter id="shadowM">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#000" flood-opacity="0.5"/>
    </filter>
  </defs>
  <rect width="600" height="450" fill="url(#bgGradM)"/>
  <path d="M 80 320 Q 180 250 240 220" stroke="#854d0e" stroke-width="8" stroke-linecap="round" fill="none"/>
  <g filter="url(#shadowM)">
    <path d="M 120 280 C 180 120, 360 80, 520 130 C 440 280, 300 350, 120 280 Z" fill="url(#leafGradM)"/>
    <!-- Region 1: Anthracnose Lesions -->
    <circle cx="230" cy="230" r="28" fill="url(#lesionHaloM)"/>
    <ellipse cx="230" cy="230" rx="16" ry="12" fill="#1c0f05" />
    <!-- Region 2: Powdery Mildew Blooms -->
    <ellipse cx="400" cy="180" rx="55" ry="38" fill="url(#powderyGradM)"/>
    <ellipse cx="460" cy="150" rx="35" ry="24" fill="url(#powderyGradM)"/>
    <!-- Region 3: Bacterial Canker Spot -->
    <polygon points="310,210 340,200 350,230 320,238" fill="#713f12" stroke="#eab308" stroke-width="1.5"/>
  </g>
</svg>`;

// 6. Sooty Mold Leaf SVG
const sootyMoldSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGradSM" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#141416" />
    </linearGradient>
    <linearGradient id="leafGradSM" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#166534" />
      <stop offset="50%" stop-color="#14532d" />
      <stop offset="100%" stop-color="#064e3b" />
    </linearGradient>
    <filter id="shadowSM">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#000" flood-opacity="0.5"/>
    </filter>
  </defs>
  <rect width="600" height="450" fill="url(#bgGradSM)"/>
  <path d="M 80 320 Q 180 250 240 220" stroke="#854d0e" stroke-width="8" stroke-linecap="round" fill="none"/>
  <g filter="url(#shadowSM)">
    <path d="M 120 280 C 180 120, 360 80, 520 130 C 440 280, 300 350, 120 280 Z" fill="url(#leafGradSM)"/>
    <!-- Sooty Black Crust -->
    <path d="M 200 240 C 260 160, 380 140, 480 160 C 420 260, 320 300, 200 240 Z" fill="#090a0f" opacity="0.85"/>
  </g>
</svg>`;

// 7. Gall Midge Leaf SVG
const gallMidgeSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGradGM" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e180d" />
    </linearGradient>
    <linearGradient id="leafGradGM" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22c55e" />
      <stop offset="50%" stop-color="#15803d" />
      <stop offset="100%" stop-color="#0f3d1e" />
    </linearGradient>
    <filter id="shadowGM">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#000" flood-opacity="0.5"/>
    </filter>
  </defs>
  <rect width="600" height="450" fill="url(#bgGradGM)"/>
  <path d="M 80 320 Q 180 250 240 220" stroke="#854d0e" stroke-width="8" stroke-linecap="round" fill="none"/>
  <g filter="url(#shadowGM)">
    <path d="M 120 280 C 180 120, 360 80, 520 130 C 440 280, 300 350, 120 280 Z" fill="url(#leafGradGM)"/>
    <!-- Wart-like blister galls -->
    <circle cx="240" cy="220" r="10" fill="#a16207" stroke="#eab308" stroke-width="1.5"/>
    <circle cx="280" cy="180" r="12" fill="#a16207" stroke="#eab308" stroke-width="1.5"/>
    <circle cx="330" cy="230" r="11" fill="#a16207" stroke="#eab308" stroke-width="1.5"/>
    <circle cx="380" cy="170" r="13" fill="#a16207" stroke="#eab308" stroke-width="1.5"/>
    <circle cx="430" cy="210" r="10" fill="#a16207" stroke="#eab308" stroke-width="1.5"/>
    <circle cx="470" cy="160" r="9" fill="#a16207" stroke="#eab308" stroke-width="1.5"/>
  </g>
</svg>`;

// 8. Cutting Weevil Leaf SVG
const cuttingWeevilSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGradCW" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#241218" />
    </linearGradient>
    <linearGradient id="leafGradCW" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4ade80" />
      <stop offset="50%" stop-color="#16a34a" />
      <stop offset="100%" stop-color="#14532d" />
    </linearGradient>
    <filter id="shadowCW">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#000" flood-opacity="0.5"/>
    </filter>
  </defs>
  <rect width="600" height="450" fill="url(#bgGradCW)"/>
  <path d="M 80 320 Q 180 250 240 220" stroke="#854d0e" stroke-width="8" stroke-linecap="round" fill="none"/>
  <g filter="url(#shadowCW)">
    <!-- Severed leaf blade with sharp transverse scissor cut -->
    <path d="M 120 280 C 160 170, 240 130, 330 160 L 335 290 C 240 310, 180 320, 120 280 Z" fill="url(#leafGradCW)"/>
    <line x1="330" y1="160" x2="335" y2="290" stroke="#991b1b" stroke-width="3" stroke-dasharray="4 2"/>
  </g>
</svg>`;


export const SAMPLE_LEAF_IMAGES = [
  {
    id: "sample-multi-disease",
    name: "Multi-Disease Mango Leaf",
    fileName: "mango_leaf_multi_pathology_specimen_07.jpg",
    fileSize: "1.85 MB",
    fileType: "image/jpeg",
    diseaseId: "multi-disease",
    categoryLabel: "Multiple Diseases",
    thumbnail: toDataUrl(multiDiseaseSvg),
    dataUrl: toDataUrl(multiDiseaseSvg),
    description: "Specimen with multiple distinct concurrent pathologies: Anthracnose necrotic lesions + Powdery Mildew mycelium patches."
  },
  {
    id: "sample-anthracnose",
    name: "Anthracnose Sample Leaf",
    fileName: "mango_leaf_anthracnose_sample_01.jpg",
    fileSize: "1.42 MB",
    fileType: "image/jpeg",
    diseaseId: "anthracnose",
    categoryLabel: "Fungal Infection",
    thumbnail: toDataUrl(anthracnoseSvg),
    dataUrl: toDataUrl(anthracnoseSvg),
    description: "Mango leaf demonstrating distinct dark brown necrotic spots with yellow chlorotic margins."
  },
  {
    id: "sample-healthy",
    name: "Healthy Sample Leaf",
    fileName: "mango_leaf_healthy_specimen_08.png",
    fileSize: "1.18 MB",
    fileType: "image/png",
    diseaseId: "healthy",
    categoryLabel: "Healthy Specimen",
    thumbnail: toDataUrl(healthySvg),
    dataUrl: toDataUrl(healthySvg),
    description: "Vibrant, unblemished mango leaf with intact photosynthetic blade and clear laminar venation."
  },
  {
    id: "sample-bacterial",
    name: "Bacterial Canker Sample Leaf",
    fileName: "mango_leaf_bacterial_canker_04.jpg",
    fileSize: "1.65 MB",
    fileType: "image/jpeg",
    diseaseId: "bacterial-canker",
    categoryLabel: "Bacterial Pathology",
    thumbnail: toDataUrl(bacterialCankerSvg),
    dataUrl: toDataUrl(bacterialCankerSvg),
    description: "Leaf exhibiting angular water-soaked lesions bounded by veins and raised corky margins."
  },
  {
    id: "sample-powdery",
    name: "Powdery Mildew Sample Leaf",
    fileName: "mango_leaf_powdery_mildew_02.jpg",
    fileSize: "1.33 MB",
    fileType: "image/jpeg",
    diseaseId: "powdery-mildew",
    categoryLabel: "Fungal Superficial",
    thumbnail: toDataUrl(powderyMildewSvg),
    dataUrl: toDataUrl(powderyMildewSvg),
    description: "Young leaf with conspicuous whitish-grey powdery mycelial patches on the upper surface."
  },
  {
    id: "sample-gall-midge",
    name: "Gall Midge Sample Leaf",
    fileName: "mango_leaf_gall_midge_05.jpg",
    fileSize: "1.52 MB",
    fileType: "image/jpeg",
    diseaseId: "gall-midge",
    categoryLabel: "Pest Infestation",
    thumbnail: toDataUrl(gallMidgeSvg),
    dataUrl: toDataUrl(gallMidgeSvg),
    description: "Leaf lamina covered with characteristic conical and wart-like blister galls caused by midge larvae."
  },
  {
    id: "sample-sooty-mold",
    name: "Sooty Mold Sample Leaf",
    fileName: "mango_leaf_sooty_mold_03.jpg",
    fileSize: "1.48 MB",
    fileType: "image/jpeg",
    diseaseId: "sooty-mold",
    categoryLabel: "Fungal Coating",
    thumbnail: toDataUrl(sootyMoldSvg),
    dataUrl: toDataUrl(sootyMoldSvg),
    description: "Dense velvety black fungal mycelial crust covering leaf surface feeding on insect honeydew."
  },
  {
    id: "sample-cutting-weevil",
    name: "Cutting Weevil Sample Leaf",
    fileName: "mango_leaf_cutting_weevil_06.jpg",
    fileSize: "1.25 MB",
    fileType: "image/jpeg",
    diseaseId: "cutting-weevil",
    categoryLabel: "Pest Damage",
    thumbnail: toDataUrl(cuttingWeevilSvg),
    dataUrl: toDataUrl(cuttingWeevilSvg),
    description: "Tender leaf with clean scissor-like transverse cut across the blade by female weevil."
  }
];

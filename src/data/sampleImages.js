/**
 * Built-in Curated Sample Mango Leaf Images & Specimens
 * Features the multi-pathology specimen from the video recording along with 8 disease classes.
 */

// SVG-encoded data URLs representing realistic disease symptoms
const createLeafSVG = ({ bgGradient, spots, leafDetails, label }) => {
  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#021b11" />
      <stop offset="100%" stop-color="#09291b" />
    </linearGradient>
    <linearGradient id="leafGrad" x1="20%" y1="0%" x2="80%" y2="100%">
      ${bgGradient}
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="16" stdDeviation="24" flood-color="#000" flood-opacity="0.65" />
    </filter>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>

  <!-- Background Environment -->
  <rect width="600" height="600" fill="url(#bgGrad)" />
  <circle cx="300" cy="300" r="260" fill="#044328" opacity="0.15" />
  <circle cx="300" cy="300" r="180" fill="#10b981" opacity="0.06" />

  <!-- Main Leaf Stem -->
  <path d="M 300 520 C 302 460, 298 380, 300 120" stroke="#855422" stroke-width="8" stroke-linecap="round" fill="none" opacity="0.75" />

  <!-- Leaf Body -->
  <g filter="url(#shadow)">
    <!-- Mango Leaf Blade Contour (Elongated lanceolate shape typical of Mangifera indica) -->
    <path d="M 300 120 
             C 390 200, 430 330, 395 440 
             C 375 500, 335 520, 300 520 
             C 265 520, 225 500, 205 440 
             C 170 330, 210 200, 300 120 Z" 
          fill="url(#leafGrad)" stroke="#1b4d2e" stroke-width="2" />
    
    <!-- Central Main Midrib Vein -->
    <path d="M 300 520 Q 302 320 300 120" stroke="#7ac989" stroke-width="4.5" stroke-linecap="round" fill="none" opacity="0.7" />
    
    <!-- Lateral Secondary Veins (Pinnate venation) -->
    <g stroke="#7ac989" stroke-width="1.8" fill="none" opacity="0.45" stroke-linecap="round">
      <path d="M 300 480 Q 340 460 365 445" />
      <path d="M 300 480 Q 260 460 235 445" />
      <path d="M 300 440 Q 350 415 385 390" />
      <path d="M 300 440 Q 250 415 215 390" />
      <path d="M 300 395 Q 360 365 395 330" />
      <path d="M 300 395 Q 240 365 205 330" />
      <path d="M 300 345 Q 360 310 390 270" />
      <path d="M 300 345 Q 240 310 210 270" />
      <path d="M 300 290 Q 355 250 375 210" />
      <path d="M 300 290 Q 245 250 225 210" />
      <path d="M 300 230 Q 340 190 350 160" />
      <path d="M 300 230 Q 260 190 250 160" />
    </g>

    <!-- Specific Pathology Visual Effects -->
    ${spots || ""}
    ${leafDetails || ""}
  </g>

  <!-- Tag Label Overlay -->
  <rect x="24" y="24" width="220" height="34" rx="17" fill="rgba(15, 23, 42, 0.85)" stroke="rgba(255, 255, 255, 0.15)" stroke-width="1" />
  <circle cx="42" cy="41" r="5" fill="#10b981" />
  <text x="56" y="46" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="13" font-weight="600" letter-spacing="0.5">SAMPLE: ${label.toUpperCase()}</text>
</svg>
  `.trim();

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
};

export const SAMPLE_LEAF_IMAGES = [
  {
    id: "sample-multi-disease",
    name: "Multi-Disease Specimen (Video Test)",
    expectedDisease: "Sooty Mold + Powdery Mildew + Gall Midge + Die Back",
    isMultiPathology: true,
    fileSize: "2.23 MB",
    fileType: "image/png",
    fileName: "f90396a2-621c-4772-9312-5a5c216b8930.png",
    previewUrl: "/samples/multi_disease_leaf.png",
    tag: "Featured Demo"
  },
  {
    id: "sample-anthracnose",
    name: "Anthracnose Specimen",
    expectedDisease: "Anthracnose",
    isMultiPathology: false,
    fileSize: "1.4 MB",
    fileType: "image/jpeg",
    fileName: "mango_leaf_anthracnose_04.jpg",
    previewUrl: createLeafSVG({
      bgGradient: `<stop offset="0%" stop-color="#1e5c33" /><stop offset="60%" stop-color="#144626" /><stop offset="100%" stop-color="#0e331b" />`,
      spots: `
        <g fill="#24140e" stroke="#100805" stroke-width="1.5">
          <ellipse cx="270" cy="260" rx="26" ry="18" transform="rotate(-15 270 260)" fill="#1c0e07" />
          <ellipse cx="340" cy="310" rx="34" ry="22" transform="rotate(25 340 310)" fill="#23120a" />
          <ellipse cx="250" cy="380" rx="28" ry="16" transform="rotate(10 250 380)" fill="#1c0e07" />
          <ellipse cx="330" cy="210" rx="18" ry="12" fill="#2d170d" />
          <ellipse cx="365" cy="380" rx="22" ry="15" fill="#1f0f08" />
          <ellipse cx="342" cy="312" rx="12" ry="7" fill="#030712" />
        </g>
        <path d="M 390 270 C 398 330, 390 380, 365 440" stroke="#3b1b0e" stroke-width="12" fill="none" opacity="0.8" stroke-linecap="round" />
      `,
      label: "Anthracnose"
    })
  },
  {
    id: "sample-powdery-mildew",
    name: "Powdery Mildew Specimen",
    expectedDisease: "Powdery Mildew",
    isMultiPathology: false,
    fileSize: "1.3 MB",
    fileType: "image/png",
    fileName: "mango_leaf_powdery_mildew_02.png",
    previewUrl: createLeafSVG({
      bgGradient: `<stop offset="0%" stop-color="#2a6f44" /><stop offset="50%" stop-color="#1a4d2e" /><stop offset="100%" stop-color="#11331e" />`,
      spots: `
        <g fill="#f1f5f9" opacity="0.75" filter="url(#glow)">
          <ellipse cx="270" cy="220" rx="35" ry="25" transform="rotate(-20 270 220)" />
          <ellipse cx="330" cy="270" rx="42" ry="30" transform="rotate(15 330 270)" />
          <ellipse cx="290" cy="350" rx="48" ry="32" />
          <ellipse cx="250" cy="430" rx="30" ry="20" />
          <ellipse cx="340" cy="400" rx="36" ry="22" transform="rotate(-10 340 400)" />
        </g>
      `,
      label: "Powdery Mildew"
    })
  },
  {
    id: "sample-bacterial-canker",
    name: "Bacterial Canker Specimen",
    expectedDisease: "Bacterial Canker",
    isMultiPathology: false,
    fileSize: "1.6 MB",
    fileType: "image/jpeg",
    fileName: "mango_leaf_bacterial_canker_08.jpg",
    previewUrl: createLeafSVG({
      bgGradient: `<stop offset="0%" stop-color="#2d6a4f" /><stop offset="50%" stop-color="#1b4332" /><stop offset="100%" stop-color="#081c15" />`,
      spots: `
        <g>
          <circle cx="260" cy="240" r="28" fill="#eab308" opacity="0.65" filter="url(#glow)" />
          <polygon points="250,225 272,230 268,252 246,248" fill="#1c0f05" stroke="#0f0702" stroke-width="2" />
          <circle cx="340" cy="280" r="34" fill="#facc15" opacity="0.6" filter="url(#glow)" />
          <polygon points="325,265 358,270 352,298 322,290" fill="#180b03" stroke="#000" stroke-width="2" />
          <circle cx="280" cy="360" r="26" fill="#eab308" opacity="0.7" filter="url(#glow)" />
          <polygon points="268,348 294,352 290,374 270,370" fill="#231006" />
        </g>
      `,
      label: "Bacterial Canker"
    })
  },
  {
    id: "sample-sooty-mold",
    name: "Sooty Mold Specimen",
    expectedDisease: "Sooty Mold",
    isMultiPathology: false,
    fileSize: "1.5 MB",
    fileType: "image/jpeg",
    fileName: "mango_leaf_sooty_mold_15.jpg",
    previewUrl: createLeafSVG({
      bgGradient: `<stop offset="0%" stop-color="#1b4332" /><stop offset="100%" stop-color="#0f291e" />`,
      spots: `
        <g fill="#09090b" opacity="0.88">
          <path d="M 240 200 C 360 180, 390 320, 370 420 C 350 490, 260 480, 230 400 C 200 320, 200 230, 240 200 Z" />
          <ellipse cx="290" cy="280" rx="65" ry="50" fill="#000" opacity="0.95" />
          <ellipse cx="320" cy="360" rx="55" ry="45" fill="#050505" opacity="0.9" />
        </g>
      `,
      label: "Sooty Mold"
    })
  },
  {
    id: "sample-die-back",
    name: "Die Back Specimen",
    expectedDisease: "Die Back",
    isMultiPathology: false,
    fileSize: "1.7 MB",
    fileType: "image/jpeg",
    fileName: "mango_leaf_die_back_03.jpg",
    previewUrl: createLeafSVG({
      bgGradient: `<stop offset="0%" stop-color="#78350f" /><stop offset="40%" stop-color="#92400e" /><stop offset="70%" stop-color="#3f6212" /><stop offset="100%" stop-color="#14532d" />`,
      spots: `
        <path d="M 300 120 C 370 180, 390 260, 350 310 C 320 340, 280 340, 240 300 C 210 250, 230 180, 300 120 Z" 
              fill="#451a03" stroke="#291002" stroke-width="2" />
      `,
      label: "Die Back"
    })
  },
  {
    id: "sample-gall-midge",
    name: "Gall Midge Specimen",
    expectedDisease: "Gall Midge",
    isMultiPathology: false,
    fileSize: "1.4 MB",
    fileType: "image/jpeg",
    fileName: "mango_leaf_gall_midge_06.jpg",
    previewUrl: createLeafSVG({
      bgGradient: `<stop offset="0%" stop-color="#2d6a4f" /><stop offset="100%" stop-color="#144626" />`,
      spots: `
        <g fill="#eab308" stroke="#a16207" stroke-width="1.5">
          <circle cx="260" cy="220" r="7" />
          <circle cx="280" cy="240" r="8" />
          <circle cx="340" cy="210" r="6" />
          <circle cx="330" cy="280" r="9" />
          <circle cx="250" cy="330" r="8" />
          <circle cx="290" cy="380" r="7" />
          <circle cx="350" cy="360" r="9" />
          <circle cx="270" cy="440" r="6" />
          <circle cx="330" cy="430" r="8" />
        </g>
      `,
      label: "Gall Midge"
    })
  },
  {
    id: "sample-healthy",
    name: "Healthy Mango Leaf",
    expectedDisease: "Healthy",
    isMultiPathology: false,
    fileSize: "1.2 MB",
    fileType: "image/png",
    fileName: "mango_leaf_healthy_12.png",
    previewUrl: createLeafSVG({
      bgGradient: `<stop offset="0%" stop-color="#22c55e" /><stop offset="40%" stop-color="#16a34a" /><stop offset="100%" stop-color="#15803d" />`,
      spots: `
        <path d="M 280 180 Q 300 320 285 460" stroke="#86efac" stroke-width="16" fill="none" opacity="0.2" stroke-linecap="round" filter="url(#glow)" />
      `,
      label: "Healthy Leaf"
    })
  }
];

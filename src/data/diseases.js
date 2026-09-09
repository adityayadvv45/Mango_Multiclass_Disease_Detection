/**
 * Configurable Mango Leaf Disease Metadata
 * This file is completely decoupled and can be updated when the trained ML model classes are modified.
 */

export const DISEASE_CLASSES = [
  {
    id: "healthy",
    name: "Healthy",
    scientificName: "Mangifera indica (Healthy)",
    category: "Healthy",
    categoryType: "healthy",
    severity: "None",
    riskLevel: "None",
    color: "emerald",
    shortDescription: "The leaf appears healthy with normal green pigmentation, uniform surface texture, and no visible pathological lesions.",
    fullDescription: "Healthy mango leaves exhibit a vibrant, uniform green coloration with well-defined venation and smooth laminar surfaces. There are no signs of necrotic lesions, fungal spores, or bacterial exudate.",
    visualIndicators: [
      "Uniform vibrant green coloration across lamina",
      "Intact leaf margins without necrosis or ragged tears",
      "Clean, unblemished upper and lower leaf surfaces",
      "Firm leaf turgidity and active photosynthetic tissue"
    ],
    recommendedActions: [
      "Maintain regular drip or basin irrigation schedule without over-watering",
      "Continue routine seasonal micronutrient and nitrogen balance application",
      "Perform periodic canopy inspections to maintain good sunlight penetration",
      "Keep orchard floor clear of fallen debris to prevent spore reservoirs"
    ],
    badgeVariant: "success"
  },
  {
    id: "anthracnose",
    name: "Anthracnose",
    scientificName: "Colletotrichum gloeosporioides",
    category: "Fungal",
    categoryType: "fungal",
    severity: "Moderate to High",
    riskLevel: "Moderate",
    color: "amber",
    shortDescription: "Fungal disease characterized by irregular dark brown to black necrotic spots with chlorotic yellow halos.",
    fullDescription: "Anthracnose is one of the most widespread fungal infections affecting mango trees worldwide. It thrives in warm, humid climates with frequent precipitation. The fungus penetrates young leaf tissues, causing shot-hole symptoms, premature leaf drop, and blossom blight.",
    visualIndicators: [
      "Dark brown to black irregular necrotic spots on leaf surface",
      "Spots may coalesce into large blighted areas on the lamina",
      "Distinct yellow chlorotic halos encircling dark lesions",
      "Brittle, shot-hole perforations as dead tissue drops out"
    ],
    recommendedActions: [
      "Prune affected twigs and dead foliage to enhance air circulation within the canopy",
      "Apply protective copper-based or approved broad-spectrum fungicide sprays during wet flushes",
      "Avoid overhead sprinkler irrigation that splashes fungal conidia across foliage",
      "Collect and safely dispose of fallen infected leaves away from the orchard"
    ],
    badgeVariant: "warning"
  },
  {
    id: "bacterial-canker",
    name: "Bacterial Canker",
    scientificName: "Xanthomonas citri pv. mangiferaeindicae",
    category: "Bacterial",
    categoryType: "bacterial",
    severity: "High",
    riskLevel: "High",
    color: "rose",
    shortDescription: "Bacterial disease causing water-soaked angular lesions that turn dark brown with raised margins and gummy exudate.",
    fullDescription: "Bacterial Canker (also known as bacterial black spot) affects leaves, twigs, and fruit. The pathogen enters through natural stomata or mechanical wounds caused by wind or insects, producing angular lesions bordered by leaf veins and severe defoliation during rainy seasons.",
    visualIndicators: [
      "Angular, water-soaked spots restricted by veins",
      "Lesions turn dark brown to black with raised, corky margins",
      "Occasional yellow translucent halo encircling early lesions",
      "Cracking of leaf tissue and premature leaf detachment"
    ],
    recommendedActions: [
      "Establish windbreaks to minimize mechanical leaf damage from windblown sand and debris",
      "Apply preventive copper oxychloride or bactericidal sprays during new flush periods",
      "Disinfect pruning tools with 70% alcohol or 10% sodium hypochlorite between trees",
      "Consult local agricultural extension officers for resistant cultivar recommendations"
    ],
    badgeVariant: "danger"
  },
  {
    id: "powdery-mildew",
    name: "Powdery Mildew",
    scientificName: "Oidium mangiferae (Pseudoidium anacardii)",
    category: "Fungal",
    categoryType: "fungal",
    severity: "Moderate",
    riskLevel: "Moderate",
    color: "purple",
    shortDescription: "Fungal infection presenting as superficial white powdery mycelial patches on young leaves and floral panicles.",
    fullDescription: "Powdery Mildew is prominent during dry, cool weather with high relative humidity. It primarily attacks emerging leaves, flowers, and tender shoots. The white powdery coating reduces photosynthesis, causing leaf curling, distortion, and significant inflorescence damage.",
    visualIndicators: [
      "White or grayish powdery fungal patches on upper/lower surfaces",
      "Curling, crinkling, and distortion of young tender leaves",
      "Purplish-brown discolored lesions beneath the powdery layer as it ages",
      "Stunted leaf development and premature shedding of infected shoots"
    ],
    recommendedActions: [
      "Apply wettable sulfur or systemic fungicides at first sign of new panicle and leaf flush",
      "Maintain canopy openness by selective pruning to decrease ambient microclimate humidity",
      "Monitor temperature shifts and morning humidity levels in early spring",
      "Ensure balanced potassium and phosphorus fertilization to strengthen leaf cuticle"
    ],
    badgeVariant: "warning"
  },
  {
    id: "sooty-mold",
    name: "Sooty Mold",
    scientificName: "Meliola mangiferae / Capnodium spp.",
    category: "Fungal / Secondary",
    categoryType: "fungal",
    severity: "Low to Moderate",
    riskLevel: "Low",
    color: "slate",
    shortDescription: "Dark, velvety black fungal crust coating leaf surfaces, feeding on insect honeydew secretions without directly penetrating tissues.",
    fullDescription: "Sooty Mold is a saprophytic fungus that grows exclusively on the sugary honeydew excreted by sap-sucking pests like mango hoppers, mealybugs, and scale insects. While non-parasitic, the dense black layer blocks sunlight, impairing photosynthesis and weakening tree vigor.",
    visualIndicators: [
      "Velvety, charcoal-black superficial layer covering leaf blades",
      "Coating can be peeled or rubbed off, revealing underlying intact green tissue",
      "Sticky honeydew residue noticeable on surrounding foliage and stems",
      "Presence of insect pests (hoppers, scales, aphids) on the lower surface"
    ],
    recommendedActions: [
      "Target the underlying insect vector (hoppers/scales) with biological or neem-based sprays",
      "Spray 2-3% starch or dilute detergent solution to cause dry mold crust to flake off",
      "Prune dense inner branches to discourage pest harboring and improve ventilation",
      "Regularly wash foliage with clean water pressure jets in dry spells"
    ],
    badgeVariant: "info"
  },
  {
    id: "die-back",
    name: "Die Back",
    scientificName: "Lasiodiplodia theobromae (Botryodiplodia theobromae)",
    category: "Physiological / Fungal",
    categoryType: "fungal",
    severity: "High",
    riskLevel: "High",
    color: "orange",
    shortDescription: "Vascular condition characterized by drying, discoloration, and upward/downward progressive dieback of leaves and twigs.",
    fullDescription: "Die Back affects vascular bundles in twigs and branches. The infection leads to drying of leaves from the apex downwards, giving an appearance of scorching. Leaves turn brown, roll inward, and remain attached to dead twigs long after drying out.",
    visualIndicators: [
      "Progressive drying, browning, and scorching starting from leaf margins and tips",
      "Dry leaves wither, roll inward, and cling tenaciously to dead twigs",
      "Dark discoloration in vascular tissue when twigs are split longitudinally",
      "Twig dieback advancing downward from terminal branches towards main limbs"
    ],
    recommendedActions: [
      "Prune affected branches 2-3 inches below the dead zone into healthy green wood",
      "Sterilize cut wounds and paint branch stubs with copper oxychloride paste",
      "Improve soil drainage and avoid deep root injuries during field cultivation",
      "Supply adequate micronutrients (especially zinc and boron) to alleviate tree stress"
    ],
    badgeVariant: "danger"
  },
  {
    id: "gall-midge",
    name: "Gall Midge",
    scientificName: "Procontarinia matteiana",
    category: "Pest / Insect Infestation",
    categoryType: "pest",
    severity: "Moderate to High",
    riskLevel: "Moderate",
    color: "amber",
    shortDescription: "Small insect pest causing elevated wart-like pimples, galls, and blister lesions across the mango leaf blade.",
    fullDescription: "The mango leaf gall midge is a tiny dipteran fly whose larvae feed within leaf tissues. The irritation stimulates localized abnormal plant cell proliferation, forming numerous conical or circular blister-like galls on the leaves. Severe infestations reduce active photosynthetic area and cause early leaf drop.",
    visualIndicators: [
      "Numerous tiny raised wart-like or conical blister galls on leaf lamina",
      "Galls initially appear pale yellowish-green and darken to reddish-brown",
      "Exit holes visible on the underside of older mature galls",
      "Leaf curling and premature defoliation in severe infestations"
    ],
    recommendedActions: [
      "Plough orchard soil beneath the tree canopy during summer to expose pupae to solar heat and natural predators",
      "Prune and destroy heavily infested shoots before the adult midges emerge",
      "Install yellow sticky traps in the canopy to monitor and capture adult flies",
      "Apply recommended systemic or botanical neem-based insecticides during new flush emergence"
    ],
    badgeVariant: "warning"
  },
  {
    id: "cutting-weevil",
    name: "Cutting Weevil",
    scientificName: "Deporaus marginatus",
    category: "Pest / Insect Damage",
    categoryType: "pest",
    severity: "Moderate",
    riskLevel: "Moderate",
    color: "rose",
    shortDescription: "Curculionid beetle that neatly cuts across tender young leaves like a pair of scissors after laying eggs.",
    fullDescription: "The mango leaf-cutting weevil targets freshly emerged, tender pinkish-green leaves. The female weevil lays eggs in the leaf blade near the base and then cuts off the leaf lamina across the petiole with mechanical precision. The severed portion drops to the ground, serving as food for developing larvae.",
    visualIndicators: [
      "Clean, sharp scissor-like transverse cuts across tender young leaves",
      "Severed distal leaf blades scattered on the orchard ground beneath trees",
      "Short stubs of petioles remaining attached to new shoot flushes",
      "Noticeable absence of new vegetative growth during peak flushing season"
    ],
    recommendedActions: [
      "Collect and safely dispose of or burn severed leaves dropped on the ground to destroy enclosed eggs and larvae",
      "Disturb tree basin soil in dry seasons to destroy overwintering grubs and pupae",
      "Spray recommended eco-friendly insecticides or neem formulations when new leaf flushes begin",
      "Maintain active orchard monitoring during flush periods to catch early cut damage"
    ],
    badgeVariant: "warning"
  }
];

export const CATEGORIES = [
  { id: "all", label: "All Classes" },
  { id: "healthy", label: "Healthy" },
  { id: "fungal", label: "Fungal Diseases" },
  { id: "bacterial", label: "Bacterial Diseases" },
  { id: "pest", label: "Pest / Insect Damage" }
];

export const getDiseaseById = (id) => {
  return DISEASE_CLASSES.find((d) => d.id.toLowerCase() === id.toLowerCase()) || null;
};

export const getDiseaseByName = (name) => {
  return DISEASE_CLASSES.find(
    (d) => d.name.toLowerCase() === name.toLowerCase()
  ) || null;
};

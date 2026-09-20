/**
 * Built-in Curated Sample Mango Leaf Images & Specimens
 * Uses genuine foliar specimens from the MangoLeafBD dataset and multi-pathology samples.
 */

export const SAMPLE_LEAF_IMAGES = [
  {
    id: "sample-multi-disease",
    name: "Multi-Disease Complex (Co-Infection)",
    expectedDisease: "Bacterial Canker + Anthracnose + Sooty Mold",
    isMultiPathology: true,
    fileSize: "2.23 MB",
    fileType: "image/png",
    fileName: "multi_disease_leaf.png",
    previewUrl: "/samples/multi_disease_leaf.png",
    tag: "Multi-Pathology"
  },
  {
    id: "sample-anthracnose",
    name: "Anthracnose Specimen",
    expectedDisease: "Anthracnose",
    isMultiPathology: false,
    fileSize: "1.42 MB",
    fileType: "image/jpeg",
    fileName: "anthracnose.jpg",
    previewUrl: "/samples/anthracnose.jpg",
    tag: "Fungal Necrosis"
  },
  {
    id: "sample-bacterial-canker",
    name: "Bacterial Canker Specimen",
    expectedDisease: "Bacterial Canker",
    isMultiPathology: false,
    fileSize: "1.58 MB",
    fileType: "image/jpeg",
    fileName: "bacterial_canker.jpg",
    previewUrl: "/samples/bacterial_canker.jpg",
    tag: "Bacterial Halo"
  },
  {
    id: "sample-cutting-weevil",
    name: "Cutting Weevil Specimen",
    expectedDisease: "Cutting Weevil",
    isMultiPathology: false,
    fileSize: "1.35 MB",
    fileType: "image/jpeg",
    fileName: "cutting_weevil.jpg",
    previewUrl: "/samples/cutting_weevil.jpg",
    tag: "Pest Damage"
  },
  {
    id: "sample-die-back",
    name: "Die Back Specimen",
    expectedDisease: "Die Back",
    isMultiPathology: false,
    fileSize: "1.65 MB",
    fileType: "image/jpeg",
    fileName: "die_back.jpg",
    previewUrl: "/samples/die_back.jpg",
    tag: "Vascular Blight"
  },
  {
    id: "sample-gall-midge",
    name: "Gall Midge Specimen",
    expectedDisease: "Gall Midge",
    isMultiPathology: false,
    fileSize: "1.48 MB",
    fileType: "image/jpeg",
    fileName: "gall_midge.jpg",
    previewUrl: "/samples/gall_midge.jpg",
    tag: "Insect Galls"
  },
  {
    id: "sample-powdery-mildew",
    name: "Powdery Mildew Specimen",
    expectedDisease: "Powdery Mildew",
    isMultiPathology: false,
    fileSize: "1.31 MB",
    fileType: "image/jpeg",
    fileName: "powdery_mildew.jpg",
    previewUrl: "/samples/powdery_mildew.jpg",
    tag: "Fungal Coating"
  },
  {
    id: "sample-sooty-mold",
    name: "Sooty Mold Specimen",
    expectedDisease: "Sooty Mold",
    isMultiPathology: false,
    fileSize: "1.52 MB",
    fileType: "image/jpeg",
    fileName: "sooty_mold.jpg",
    previewUrl: "/samples/sooty_mold.jpg",
    tag: "Saprophytic Fungi"
  },
  {
    id: "sample-healthy",
    name: "Healthy Mango Leaf",
    expectedDisease: "Healthy",
    isMultiPathology: false,
    fileSize: "1.25 MB",
    fileType: "image/jpeg",
    fileName: "healthy.jpg",
    previewUrl: "/samples/healthy.jpg",
    tag: "Optimal Foliage"
  }
];

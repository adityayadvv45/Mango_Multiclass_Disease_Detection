"""
Neural Network Architectures, Class Metadata & Preprocessing for Mango Leaf Disease Classification.
Supports: EfficientNet-B0, MobileNetV3-Large, ResNet18.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import List, Dict, Tuple, Optional

# Standard ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

# The canonical 8 botanical classes
CANONICAL_CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mold"
]

# Aliases to handle dataset naming variations (e.g., Sooty Mould vs Sooty Mold)
CLASS_ALIASES = {
    "sooty mould": "Sooty Mold",
    "sooty mold": "Sooty Mold",
    "healthy": "Healthy",
    "anthracnose": "Anthracnose",
    "bacterial canker": "Bacterial Canker",
    "cutting weevil": "Cutting Weevil",
    "die back": "Die Back",
    "dieback": "Die Back",
    "gall midge": "Gall Midge",
    "powdery mildew": "Powdery Mildew"
}

def normalize_class_name(name: str) -> str:
    cleaned = name.strip().lower()
    return CLASS_ALIASES.get(cleaned, name)

# Comprehensive botanical metadata dictionary
DISEASE_METADATA: Dict[str, Dict] = {
    "Healthy": {
        "scientificName": "Mangifera indica (Healthy Foliage)",
        "category": "Healthy",
        "status": "Healthy Foliage",
        "risk": "None",
        "riskColor": "emerald",
        "badgeBg": "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
        "pathogen": "None (Physiologically Normal)",
        "severityLevel": "0 / 5 (Optimal)",
        "shortDescription": "Leaf exhibits vibrant green coloration, intact cuticle, and no visible pathogen infestation.",
        "description": "The leaf displays uniform chlorophyll distribution, intact leaf margins, and robust cellular structure without necrotic lesions, bacterial ooze, or fungal mycelium.",
        "causes": "Optimal orchard conditions with adequate sunlight, well-balanced NPK nutrition, properly aerated soil, and absence of pathogenic fungal spores or sap-sucking pests.",
        "visualIndicators": [
            "Uniform deep green pigmentation across leaf blade",
            "Smooth, undamaged waxy cuticle with distinct clear vein patterns",
            "Absence of necrotic dark spots, yellow halos, or powdery coatings",
            "Well-formed leaf margin without curling, distortion, or wilting"
        ],
        "recommendedSteps": [
            "Maintain standard orchard maintenance with balanced seasonal NPK fertilization (8:4:8 ratio).",
            "Ensure regular drip irrigation scheduling to prevent drought stress during flowering and fruit set.",
            "Conduct routine canopy pruning to facilitate maximum sunlight penetration and air circulation.",
            "Periodically monitor lower foliage surfaces for early signs of opportunistic pest or spore buildup."
        ]
    },
    "Anthracnose": {
        "scientificName": "Colletotrichum gloeosporioides",
        "category": "Fungal",
        "status": "Disease Detected",
        "risk": "High",
        "riskColor": "rose",
        "badgeBg": "bg-rose-500/10 text-rose-400 border-rose-500/30",
        "pathogen": "Fungus (Colletotrichum gloeosporioides)",
        "severityLevel": "4 / 5 (Severe)",
        "shortDescription": "A widespread destructive fungal disease causing dark brown to black necrotic lesions on leaves and shoots.",
        "description": "Anthracnose is one of the most severe foliar infections in tropical mango orchards. Caused by Colletotrichum gloeosporioides, spores germinate rapidly during humid spells, penetrating leaf cuticles directly through appressoria and stomata to produce coalescing necrotic spots.",
        "causes": "Prolonged leaf wetness (>95% relative humidity), warm temperatures (25–30°C), and rain splashing that disperses conidiospores from dormant twigs or mummified floral residues.",
        "visualIndicators": [
            "Dark brown to black circular or angular necrotic spots",
            "Lesions coalescing into large scorched, ragged dead patches",
            "Brittle necrotic centers that dry and drop out ('shot-hole' effect)",
            "Browning, withering, and curling along leaf margins and tips"
        ],
        "recommendedSteps": [
            "Prune infected twigs 2–3 inches below lesions and safely incinerate all fallen foliage.",
            "Apply protective copper oxychloride (0.3% / 3g per liter) or azoxystrobin spray at early leaf flush.",
            "Ensure open tree canopy architecture through summer thinning to accelerate foliage drying after rain.",
            "Rotate with systemic mancozeb or difenoconazole fungicides during persistent monsoon humidity."
        ]
    },
    "Bacterial Canker": {
        "scientificName": "Xanthomonas campestris pv. mangiferaeindicae",
        "category": "Bacterial",
        "status": "Disease Detected",
        "risk": "High",
        "riskColor": "amber",
        "badgeBg": "bg-amber-500/10 text-amber-400 border-amber-500/30",
        "pathogen": "Bacterium (Xanthomonas campestris pv. mangiferaeindicae)",
        "severityLevel": "4 / 5 (Severe)",
        "shortDescription": "Bacterial infection causing angular water-soaked lesions with prominent bright yellow chlorotic halos.",
        "description": "Bacterial canker is a destructive bacterial disease causing raised, water-soaked angular lesions bounded by leaf veins, surrounded by bright yellow chlorotic halos. In advanced stages, bacterial exudate hardens into a brownish-black crust.",
        "causes": "Xanthomonas bacteria invade foliage via stomatal apertures and mechanical abrasions created by storms, insects, or unsanitized harvesting shears under warm, humid conditions.",
        "visualIndicators": [
            "Angular, water-soaked dark brown to black lesions bounded by veins",
            "Prominent bright yellow chlorotic halos surrounding dark spots",
            "Elevated, raised blister-like spots that exude sticky bacterial ooze",
            "Cracking and necrotic browning of midribs and secondary veins"
        ],
        "recommendedSteps": [
            "Avoid overhead sprinkler irrigation to stop waterborne splash dissemination of bacteria.",
            "Apply bactericidal formulation (Streptocycline 100 ppm + Copper Oxychloride 0.2%) during vegetative flush.",
            "Prune and destroy heavily blighted shoots using shears disinfected with 70% alcohol or 1% bleach.",
            "Establish orchard windbreaks to minimize leaf tearing and wind-driven rain abrasion."
        ]
    },
    "Powdery Mildew": {
        "scientificName": "Oidium mangiferae (Pseudoidium anacardii)",
        "category": "Fungal",
        "status": "Disease Detected",
        "risk": "Moderate",
        "riskColor": "purple",
        "badgeBg": "bg-purple-500/10 text-purple-400 border-purple-500/30",
        "pathogen": "Fungus (Oidium mangiferae / Pseudoidium anacardii)",
        "severityLevel": "3 / 5 (Moderate)",
        "shortDescription": "Superficial white powdery fungal mycelium covering the leaf surface, causing curling and distortion.",
        "description": "Powdery mildew attacks tender emerging flushes, inflorescences, and young leaves. Characterized by white to grayish powdery patches of airborne conidia and fungal mycelium, causing leaf distortion, chlorosis, and premature leaf shedding.",
        "causes": "Windborne airborne conidia thriving in cool, dry nights (10–15°C) followed by warm, dry days (25–30°C) with moderate morning humidity.",
        "visualIndicators": [
            "White or grayish powdery patches covering upper and lower surfaces",
            "Curling, crinkling, and upward distortion of young tender leaves",
            "Purplish-brown necrotic discoloration underneath powdery fungal layers",
            "Stunted foliar development and shedding of infected shoots"
        ],
        "recommendedSteps": [
            "Apply wettable sulfur (0.2% / 2g per liter) or systemic fungicides (Hexaconazole 0.1%) at first sign of flush.",
            "Maintain canopy openness by selective pruning to decrease ambient microclimate humidity.",
            "Monitor temperature shifts and morning humidity levels in early spring flowering periods.",
            "Ensure balanced potassium and phosphorus fertilization to strengthen leaf cuticle resistance."
        ]
    },
    "Sooty Mold": {
        "scientificName": "Meliola mangiferae / Capnodium spp.",
        "category": "Fungal",
        "status": "Disease Detected",
        "risk": "Moderate",
        "riskColor": "slate",
        "badgeBg": "bg-slate-400/10 text-slate-300 border-slate-500/30",
        "pathogen": "Saprophytic Fungi (Capnodium mangiferae / Meliola spp.)",
        "severityLevel": "2 / 5 (Moderate-Low)",
        "shortDescription": "Velvety black fungal coating covering leaf surfaces, secondary to honeydew-secreting insect pests.",
        "description": "Sooty mold is a superficial saprophytic fungus that feeds exclusively on sticky honeydew secretions left by sap-sucking insects (mango hoppers, mealybugs, scale insects). The dense charcoal layer blocks sunlight, severely reducing leaf photosynthetic capability.",
        "causes": "Heavy infestations of honeydew-secreting phloem-feeding pests (Idiocerus mango hoppers, Rastrococcus mealybugs) in dense, poorly ventilated orchards.",
        "visualIndicators": [
            "Velvety, charcoal-black superficial layer covering leaf blades",
            "Coating can be peeled or rubbed off, revealing underlying intact green tissue",
            "Sticky honeydew residue noticeable on surrounding foliage and stems",
            "Presence of insect pests (hoppers, scales, aphids) on the lower leaf surface"
        ],
        "recommendedSteps": [
            "Target the underlying insect vector (hoppers/scales) with biological or neem-based sprays (Azadirachtin 0.03%).",
            "Spray 2–3% starch or dilute detergent solution to cause dry mold crust to flake off in the sun.",
            "Prune dense inner branches to discourage pest harboring and improve ventilation.",
            "Regularly wash foliage with clean water pressure jets in dry spells to eliminate honeydew build-up."
        ]
    },
    "Die Back": {
        "scientificName": "Lasiodiplodia theobromae (Botryodiplodia theobromae)",
        "category": "Physiological / Fungal",
        "status": "Disease Detected",
        "risk": "High",
        "riskColor": "orange",
        "badgeBg": "bg-orange-500/10 text-orange-400 border-orange-500/30",
        "pathogen": "Fungus (Lasiodiplodia theobromae)",
        "severityLevel": "4 / 5 (Severe)",
        "shortDescription": "Vascular pathogen causing progressive drying and browning of leaves and twigs from the apex downwards.",
        "description": "Die Back affects vascular bundles in twigs and branches. The infection leads to drying of leaves from the apex downwards, giving an appearance of scorching. Leaves turn brown, roll inward, and remain attached to dead twigs long after drying out.",
        "causes": "Lasiodiplodia fungal invasion through mechanical wounds, frost injury, root damage, or severe drought stress, blocking vascular xylem vessels.",
        "visualIndicators": [
            "Progressive drying, browning, and scorching starting from leaf margins and tips",
            "Dry leaves wither, roll inward, and cling tenaciously to dead twigs",
            "Dark discoloration in vascular tissue when twigs are split longitudinally",
            "Twig dieback advancing downward from terminal branches towards main limbs"
        ],
        "recommendedSteps": [
            "Prune affected branches 2–3 inches below the dead zone into healthy green wood.",
            "Sterilize cut wounds and paint branch stubs with copper oxychloride paste (Bordeaux paste).",
            "Improve soil drainage and avoid deep root injuries during field cultivation.",
            "Supply adequate micronutrients (especially zinc and boron) to alleviate tree physiological stress."
        ]
    },
    "Gall Midge": {
        "scientificName": "Procontarinia matteiana (Diptera: Cecidomyiidae)",
        "category": "Pest / Entomological",
        "status": "Disease Detected",
        "risk": "High",
        "riskColor": "amber",
        "badgeBg": "bg-yellow-500/10 text-yellow-400 border-yellow-500/30",
        "pathogen": "Insect Pest (Procontarinia matteiana)",
        "severityLevel": "3 / 5 (Moderate-High)",
        "shortDescription": "Pest infestation resulting in wart-like galls on leaves that eventually turn necrotic and drop out.",
        "description": "Mango gall midge is a tiny insect pest whose females deposit eggs inside tender young leaf tissues. Developing larvae secrete chemicals triggering abnormal plant cellular growth, producing small wart-like elevated galls on both leaf surfaces.",
        "causes": "Infestation by Procontarinia gall midges during new foliage flush cycles in warm, still, humid microclimates.",
        "visualIndicators": [
            "Dense, raised wart-like pimple galls scattered across the leaf blade",
            "Galls initially pale yellow-green turning reddish-brown and necrotic with age",
            "Tiny exit holes visible on the underside of older dried galls",
            "Severe curling, malformation, and premature leaf drop on heavily infested shoots"
        ],
        "recommendedSteps": [
            "Collect and destroy heavily galled foliage and rake fallen leaves beneath tree canopies.",
            "Spray systemic insecticides (Imidacloprid 17.8 SL at 0.3ml/L or Dimethoate 30 EC at 1.5ml/L) during flush.",
            "Deep plow orchard basin soil around tree driplines in autumn to expose overwintering pupae to sunlight.",
            "Conserve natural parasitic wasps (Platygaster spp.) by avoiding indiscriminate broad-spectrum sprays."
        ]
    },
    "Cutting Weevil": {
        "scientificName": "Deporaus marginatus (Coleoptera: Attelabidae)",
        "category": "Pest / Entomological",
        "status": "Disease Detected",
        "risk": "Moderate",
        "riskColor": "teal",
        "badgeBg": "bg-teal-500/10 text-teal-400 border-teal-500/30",
        "pathogen": "Insect Pest (Deporaus marginatus)",
        "severityLevel": "2 / 5 (Moderate)",
        "shortDescription": "Leaf-cutting beetle pest causing clean transverse cuts across tender mango leaves.",
        "description": "The mango leaf-cutting weevil targets fresh tender flushes. Female weevils lay eggs near the base of the leaf blade and then neatly sever the apical portion with sharp mandibles, dropping the leaf tip to the ground.",
        "causes": "Active adult weevil populations emerging during spring/monsoon vegetative flush to oviposit on succulent young leaves.",
        "visualIndicators": [
            "Clean, transverse straight-line razor cuts across leaf blades",
            "Truncated, amputated leaf stubs remaining attached to twigs",
            "Accumulation of neatly severed leaf tips on the orchard floor below",
            "Stunted shoots lacking normal foliar canopy surface area"
        ],
        "recommendedSteps": [
            "Collect and incinerate severed leaf tips from orchard ground to kill developing larvae inside.",
            "Apply foliar neem oil (5ml/L) or contact insecticide (Chlorpyrifos 20 EC @ 2ml/L) during tender flush emergence.",
            "Shake infested branches gently over collection sheets in early mornings to catch sluggish adult weevils.",
            "Maintain clean orchard sanitation and eliminate weed hosts around the perimeter."
        ]
    }
}

def build_model(model_name: str, num_classes: int = 8) -> nn.Module:
    """
    Builds the deep neural network architecture.
    """
    if model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == "MobileNetV3-Large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    return model

def get_inference_transform() -> transforms.Compose:
    """
    Standard PyTorch inference transformation matching ImageNet training statistics.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def preprocess_image_for_model(image_rgb: Image.Image) -> torch.Tensor:
    """
    Prepares a PIL RGB Image into a normalized float32 tensor (1, 3, 224, 224).
    """
    transform = get_inference_transform()
    tensor = transform(image_rgb).unsqueeze(0)
    return tensor

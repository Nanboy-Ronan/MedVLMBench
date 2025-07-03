# tasks
TASKS = ["vqa", "caption", "diagnosis"]


# models
CLIP_MODELS = [
    "BLIP",
    "BLIP2-2.7b",
    "BioMedCLIP",
    "CLIP",
    "MedCLIP",
    "PMCCLIP",
    "PLIP"
]

LANGUAGE_MODELS = ["LLaVA-1.5", "LLaVA-Med", "XGenMiniV1", "XrayGPT", "NVILA", "VILA-M3", "VILA1.5"]

MODELS = CLIP_MODELS + LANGUAGE_MODELS


# datasets
VQA_DATASETS = ["SLAKE", "PathVQA", "VQA-RAD"]
CAPTION_DATASETS = ["Harvard-FairVLMed10k", "MIMIC_CXR"]
DIAGNOSIS_DATASETS = ["PneumoniaMNIST", "BreastMNIST", "DermaMNIST", "Camelyon17", "Drishti", "HAM10000", "ChestXray", "GF3300"]

DATASETS = VQA_DATASETS + CAPTION_DATASETS + DIAGNOSIS_DATASETS

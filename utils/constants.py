# tasks
TASKS = ["vqa", "caption", "diagnosis"]


# models
CLIP_MODELS = [
    "BLIP",
    "BLIP2-2.7b",
    "BioMedCLIP",
    "CLIP",
    "MedCLIP",
    "PMCCLIP"
]

LANGUAGE_MODELS = ["LLaVA-1.5", "LLaVA-Med", "XGenMiniV1", "XrayGPT"]

MODELS = CLIP_MODELS + LANGUAGE_MODELS


# datasets
VQA_DATASETS = ["SLAKE", "PathVQA", "VQA-RAD"]
CAPTION_DATASETS = ["Harvard-FairVLMed10k", "MIMIC_CXR"]
DIAGNOSIS_DATASETS = ["PneumoniaMNIST", "BreastMNIST", "DermaMNIST", "Camelyon17", "Drishti", "HAM10000"]

DATASETS = VQA_DATASETS + CAPTION_DATASETS + DIAGNOSIS_DATASETS

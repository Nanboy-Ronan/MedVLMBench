# tasks
TASKS = ["vqa", "caption"]


# models
CLIP_MODELS = [
    "BiomedCLIP",
    "PubMedCLIP",
    "MedCLIP",
    "CLIP",
    "BLIP",
    "BLIP2-2.7b",
]

LANGUAGE_MODELS = ["LLaVA-1.5", "LLaVA-Med", "XGenMiniV1"]

MODELS = CLIP_MODELS + LANGUAGE_MODELS


# ddddddddatasets
VQA_DATASETS = ["SLAKE"]

DATASETS = VQA_DATASETS

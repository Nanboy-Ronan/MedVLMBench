# tasks
TASKS = ["vqa", "caption"]


# models
CLIP_MODELS = [
    "BLIP",
    "BLIP2-2.7b",
]

LANGUAGE_MODELS = ["LLaVA-1.5", "LLaVA-Med", "XGenMiniV1", "XrayGPT"]

MODELS = CLIP_MODELS + LANGUAGE_MODELS


# ddddddddatasets
VQA_DATASETS = ["SLAKE", "PathVQA", "VQA-RAD"]
CAPTION_DATASETS = ["Harvard-FairVLMed10k"]

DATASETS = VQA_DATASETS + CAPTION_DATASETS

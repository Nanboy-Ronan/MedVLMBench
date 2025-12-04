import requests
import json
import pandas as pd
import time
import tqdm
import os
import re
from easydict import EasyDict as edict
from torchvision.transforms.functional import to_pil_image

import sys
from PIL import Image
from os.path import dirname

sys.path.append(dirname(dirname(dirname(dirname(dirname(__file__))))))

from dataset import get_dataset

root = "/research/d5/gds/yzhong22/datasets"

DATASET_ROOTS = {
    "SLAKE": os.path.join(root, "SLAKE"),
    "PathVQA": os.path.join(root, "PathVQA"),
    "VQA-RAD": os.path.join(root, "VQA-RAD"),
    "Harvard-FairVLMed10k": os.path.join(root, "Harvard-FairVLMed10k"),
    "MedXpertQA": os.path.join(root, "MedXpertQA"),
}

DATA_IMG_PATH = {
    "SLAKE": "/research/d5/gds/yzhong22/datasets/SLAKE/imgs",
    "PathVQA": "None",
    "VQA-RAD": "None",
    "Harvard-FairVLMed10k": "/research/d5/gds/yzhong22/datasets/Harvard-FairVLMed10k",
    "MedXpertQA": "/research/d5/gds/yzhong22/datasets/MedXpertQA",
}

if __name__ == "__main__":
    # datasets = ["SLAKE", "PathVQA", "VQA-RAD", "Harvard-FairVLMed10k"]
    datasets = ["MedXpertQA"]

    for dataset in datasets:
        data_args = edict(split="train", task="vqa", dataset=dataset, seed=0, image_path=DATA_IMG_PATH[dataset])

        data = get_dataset(data_args)
        json_out = []

        for idx, subject in tqdm.tqdm(enumerate(data)):
            image = subject["image"]
            qs = subject["query"]
            answer = subject["label"]
            prompt_template = subject["prompt_template"]

            image_path = subject["image_path"]

            if ";" in image_path:
                image_path_list = image_path.split(";")
            else:
                image_path_list = [image_path]
            num_images = len(image_path_list)

            conversations_json = [
                {"from": "human", "value": "<image>" * num_images + prompt_template.format(qs)},
                {"from": "gpt", "value": answer},
            ]
            images_json = []
            for idx_num, p in enumerate(image_path_list):
                if p != "NA" and (
                    str.endswith(p, ".jpg")
                    or str.endswith(p, ".png")
                    or str.endswith(p, ".jpeg")
                    or str.endswith(p, ".JPG")
                ):
                    images_json.append(p)
                else:
                    save_dir = os.path.join(DATASET_ROOTS[dataset], f"temp_images_train")
                    img_save_path = os.path.join(save_dir, f"{idx}_{idx_num}.jpg")

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    img_pil = to_pil_image(image).convert("RGB")
                    img_pil.save(img_save_path)

                    images_json.append(img_save_path)

            # multi image
            # if dataset in ["MedXpertQA"]:
            #     print(image_path)
            #     image_path_list = image_path.split(";")
            #     num_images = len(image_path_list)
            #     print(image_path_list)

            #     conversations_json = [
            #         {"from": "human", "value": "<image>" * num_images + prompt_template.format(qs)},
            #         {"from": "gpt", "value": answer},
            #     ]
            #     images_json = []
            #     for image_path in image_path_list:
            #         if image_path != "NA" and (
            #             str.endswith(image_path, ".jpg")
            #             or str.endswith(image_path, ".png")
            #             or str.endswith(image_path, ".jpeg")
            #         ):
            #             images_json.append(image_path)
            #         else:
            #             save_dir = os.path.join(DATASET_ROOTS[dataset], f"temp_images_train")
            #             img_save_path = os.path.join(save_dir, f"{idx}.jpg")

            #             if not os.path.exists(save_dir):
            #                 os.makedirs(save_dir)

            #             img_pil = to_pil_image(image).convert("RGB")
            #             img_pil.save(img_save_path)

            #             images_json.append(img_save_path)
            #     break
            # else:
            #     conversations_json = [
            #         {"from": "human", "value": "<image>" + prompt_template.format(qs)},
            #         {"from": "gpt", "value": answer},
            #     ]

            #     images_json = []

            #     if image_path != "NA" and (
            #         str.endswith(image_path, ".jpg")
            #         or str.endswith(image_path, ".png")
            #         or str.endswith(image_path, ".jpeg")
            #     ):
            #         images_json.append(image_path)
            #     else:
            #         save_dir = os.path.join(DATASET_ROOTS[dataset], f"temp_images_train")
            #         img_save_path = os.path.join(save_dir, f"{idx}.jpg")

            #         if not os.path.exists(save_dir):
            #             os.makedirs(save_dir)

            #         img_pil = to_pil_image(image).convert("RGB")
            #         img_pil.save(img_save_path)

            #         images_json.append(img_save_path)

            json_out.append({"conversations": conversations_json, "images": images_json})
            # break

        json_save_path = os.path.join(DATASET_ROOTS[dataset], "vqa_sharegpt")
        if not os.path.exists(json_save_path):
            os.makedirs(json_save_path)

        with open(os.path.join(json_save_path, "train.json"), "w") as json_file:
            json.dump(json_out, json_file, indent=4)

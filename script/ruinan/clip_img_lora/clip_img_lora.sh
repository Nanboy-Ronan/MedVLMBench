#!/bin/bash

#SBATCH --nodes=1
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --mem=48G             # memory per node
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=jinruinan@163.com
#SBATCH --mail-type=ALL
 
################################################################################

export TRANSFORMERS_CACHE=/projects/rjin-mh/rjin02/.cache/huggingface/hub
export HF_HOME=/projects/rjin-mh/rjin02/.cache/huggingface/hub

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "SLURM_PROCID="$SLURM_PROCID

export PATH=/pkgs/anaconda3/bin:$PATH
export CUDA_LAUNCH_BLOCKING=1
source activate
conda activate /h/rjin02/.conda/envs/torch2.3


export HOME=/projects/rjin-mh/rjin02

cd /projects/rjin-mh/rjin02/MedVLMBench

nvidia-smi

echo "Run started at:- "
date


# Training
# PneumoniaMNIST
# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model CLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3

# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset PneumoniaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-txt-lora --dataset PneumoniaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model PMCCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 2 \
#     --learning_rate 5e-3

# BreastMNIST
# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset BreastMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model CLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3

# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset BreastMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset BreastMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset BreastMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset BreastMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3

# HAM10000
# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset HAM10000 --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model CLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3

# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset HAM10000 --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model BLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset HAM10000 --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset HAM10000 --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset HAM10000 --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3

# Drishti
# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset Drishti --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model CLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3

# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset Drishti --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model BLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset Drishti --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset Drishti --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset Drishti --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# Camelyon17
python run_train.py \
    --task diagnosis --usage clip-img-lora --dataset Camelyon17 --split train \
    --image_path ./data/camelyon17_v1.0/patches \
    --output_dir ./log \
    --model CLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 50 \
    --learning_rate 5e-3

# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --output_dir ./log \
#     --model BLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3








# DermaMNIST
# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model CLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3

# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3


# python run_train.py \
#     --task diagnosis --usage clip-img-lora --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 50 \
#     --learning_rate 5e-3
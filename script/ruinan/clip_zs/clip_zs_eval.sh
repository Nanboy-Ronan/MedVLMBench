#!/bin/bash

#SBATCH --nodes=1
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=jinruinan@163.com
#SBATCH --mail-type=ALL
 
################################################################################

# ##### Number of total processes 
# echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
# echo "Nodelist:= " $SLURM_JOB_NODELIST
# echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
# echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
# echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# echo "SLURM_PROCID="$SLURM_PROCID


module load arrow/16.1.0 rust/1.76.0
module load opencv/4.10.0

source ~/projects/aip-xli135/nanboy/env/fairmedfm/bin/activate

nvidia-smi
cd /home/nanboy/projects/aip-xli135/nanboy/MedVLMBench

export HOME=/home/nanboy/projects/aip-xli135/nanboy

nvidia-smi


# Zeroshot
# PneumoniaMNIST (not used)
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model CLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model BLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model BioMedCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model MedCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model BLIP2-2.7b --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model PMCCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PneumoniaMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model PLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# BreastMNIST (not used)
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model CLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model BLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model BioMedCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model MedCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model BLIP2-2.7b --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model PMCCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset BreastMNIST --split test \
#     --image_path ./data \
#     --exp_path ./log \
#     --model PLIP --model_path "original_pretrained" \
#     --cache_dir ./cache



# PAPILA
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PAPILA --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model CLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache


# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PAPILA --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model BLIP2-2.7b --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PAPILA --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model SigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

    
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PAPILA --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model MedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PAPILA --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model BioMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PAPILA --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model MedSigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PAPILA --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model PubMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset PAPILA --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model PLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# HarvardFairVLMed10k
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HarvardFairVLMed10k --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model CLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache


# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HarvardFairVLMed10k --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model BLIP2-2.7b --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HarvardFairVLMed10k --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model SigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

    
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HarvardFairVLMed10k --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model MedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HarvardFairVLMed10k --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model BioMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HarvardFairVLMed10k --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model MedSigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HarvardFairVLMed10k --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model PubMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HarvardFairVLMed10k --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model PLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# HAM10000
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HAM10000 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model CLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache


# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HAM10000 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model BLIP2-2.7b --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HAM10000 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model SigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

    
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HAM10000 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model MedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HAM10000 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model BioMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HAM10000 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model MedSigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HAM10000 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model PubMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset HAM10000 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model PLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# GF3300
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset GF3300 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model CLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache


# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset GF3300 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model BLIP2-2.7b --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset GF3300 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model SigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

    
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset GF3300 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model MedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset GF3300 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model BioMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset GF3300 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model MedSigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset GF3300 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model PubMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset GF3300 --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --exp_path ./log \
#     --model PLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache


# CheXpert
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset CheXpert --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/DataSets \
#     --exp_path ./log \
#     --model CLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache


# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset CheXpert --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/DataSets \
#     --exp_path ./log \
#     --model BLIP2-2.7b --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset CheXpert --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/DataSets \
#     --exp_path ./log \
#     --model SigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

    
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset CheXpert --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/DataSets \
#     --exp_path ./log \
#     --model MedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset CheXpert --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/DataSets \
#     --exp_path ./log \
#     --model BioMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset CheXpert --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/DataSets \
#     --exp_path ./log \
#     --model MedSigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset CheXpert --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/DataSets \
#     --exp_path ./log \
#     --model PubMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset CheXpert --split test \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/DataSets \
#     --exp_path ./log \
#     --model PLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache


# Camelyon17
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --exp_path ./log \
#     --model CLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache


# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --exp_path ./log \
#     --model BLIP2-2.7b --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --exp_path ./log \
#     --model SigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

    
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --exp_path ./log \
#     --model MedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --exp_path ./log \
#     --model BioMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --exp_path ./log \
#     --model MedSigLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --exp_path ./log \
#     --model PubMedCLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Camelyon17 --split test \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --exp_path ./log \
#     --model PLIP --model_path "original_pretrained" \
#     --save_pred \
#     --cache_dir ./cache

# Drishti (not used)
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Drishti --split test \
#     --image_path /mnt/sdc/rjin02/DataSets \
#     --exp_path ./log \
#     --model CLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Drishti --split test \
#     --image_path /mnt/sdc/rjin02/DataSets \
#     --exp_path ./log \
#     --model BLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Drishti --split test \
#     --image_path /mnt/sdc/rjin02/DataSets \
#     --exp_path ./log \
#     --model BioMedCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Drishti --split test \
#     --image_path /mnt/sdc/rjin02/DataSets \
#     --exp_path ./log \
#     --model MedCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Drishti --split test \
#     --image_path /mnt/sdc/rjin02/DataSets \
#     --exp_path ./log \
#     --model BLIP2-2.7b --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Drishti --split test \
#     --image_path /mnt/sdc/rjin02/DataSets \
#     --exp_path ./log \
#     --model PMCCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset Drishti --split test \
#     --image_path /mnt/sdc/rjin02/DataSets \
#     --exp_path ./log \
#     --model PLIP --model_path "original_pretrained" \
#     --cache_dir ./cache


# ChestXray (not used)
# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset ChestXray --split test \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --exp_path ./log \
#     --model CLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset ChestXray --split test \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --exp_path ./log \
#     --model BLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset ChestXray --split test \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --exp_path ./log \
#     --model BioMedCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset ChestXray --split test \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --exp_path ./log \
#     --model MedCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset ChestXray --split test \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --exp_path ./log \
#     --model BLIP2-2.7b --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset ChestXray --split test \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --exp_path ./log \
#     --model PMCCLIP --model_path "original_pretrained" \
#     --cache_dir ./cache

# python run_eval.py \
#     --task diagnosis --usage clip-zs --dataset ChestXray --split test \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --exp_path ./log \
#     --model PLIP --model_path "original_pretrained" \
#     --cache_dir ./cache
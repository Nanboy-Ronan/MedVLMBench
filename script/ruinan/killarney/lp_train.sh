#!/bin/bash

#SBATCH --nodes=1
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:1
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

# HAM10000
# python run_train.py \
#     --task diagnosis --usage lp --dataset HAM10000 --split train \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --output_dir ./log \
#     --model CLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Ongoing

python run_train.py \
    --task diagnosis --usage lp --dataset HAM10000 --split train \
    --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
    --output_dir ./log \
    --model BLIP --model_path not_given \
    --cache_dir ./cache \
    --num_train_epochs 10 \
    --learning_rate 5e-5

# CUDA_VISIBLE_DEVICES=2 python run_train.py \
#     --task diagnosis --usage lp --dataset HAM10000 --split train \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Ongoing

# CUDA_VISIBLE_DEVICES=3 python run_train.py \
#     --task diagnosis --usage lp --dataset HAM10000 --split train \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Ongoing

# CUDA_VISIBLE_DEVICES=4 python run_train.py \
#     --task diagnosis --usage lp --dataset HAM10000 --split train \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Ongoing

# CUDA_VISIBLE_DEVICES=5 python run_train.py \
#     --task diagnosis --usage lp --dataset HAM10000 --split train \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --output_dir ./log \
#     --model PLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Done

# CUDA_VISIBLE_DEVICES=6 python run_train.py \
#     --task diagnosis --usage lp --dataset HAM10000 --split train \
#     --image_path /home/nanboy/projects/aip-xli135/nanboy/FairMedFM-DNE/data \
#     --output_dir ./log \
#     --model MedSigLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Done




# CheXpert
# CUDA_VISIBLE_DEVICES=4 python run_train.py \
#     --task diagnosis --usage lp --dataset CheXpert --split train \
#     --image_path /bigdata/rjin02/DataSets \
#     --output_dir ./log \
#     --model CLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Ongoing

# CUDA_VISIBLE_DEVICES=5 python run_train.py \
#     --task diagnosis --usage lp --dataset CheXpert --split train \
#     --image_path /bigdata/rjin02/DataSets \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Ongoing

# CUDA_VISIBLE_DEVICES=6 python run_train.py \
#     --task diagnosis --usage lp --dataset CheXpert --split train \
#     --image_path /bigdata/rjin02/DataSets \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Ongoing

# CUDA_VISIBLE_DEVICES=7 python run_train.py \
#     --task diagnosis --usage lp --dataset CheXpert --split train \
#     --image_path /bigdata/rjin02/DataSets \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Ongoing

# CUDA_VISIBLE_DEVICES=5 python run_train.py \
#     --task diagnosis --usage lp --dataset CheXpert --split train \
#     --image_path /bigdata/rjin02/DataSets \
#     --output_dir ./log \
#     --model PLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Ongoing

# CUDA_VISIBLE_DEVICES=6 python run_train.py \
#     --task diagnosis --usage lp --dataset CheXpert --split train \
#     --image_path /bigdata/rjin02/DataSets \
#     --output_dir ./log \
#     --model MedSigLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 # Ongoing





# Drishti
# python run_train.py \
#     --task diagnosis --usage lp --dataset Drishti --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model CLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# python run_train.py \
#     --task diagnosis --usage lp --dataset Drishti --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model BLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# python run_train.py \
#     --task diagnosis --usage lp --dataset Drishti --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# python run_train.py \
#     --task diagnosis --usage lp --dataset Drishti --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# python run_train.py \
#     --task diagnosis --usage lp --dataset Drishti --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5


# python run_train.py \
#     --task diagnosis --usage lp --dataset Drishti --split train \
#     --image_path ../DataSets \
#     --output_dir ./log \
#     --model PMCCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# # Camelyon17
# CUDA_VISIBLE_DEVICES=0 python run_train.py \
#     --task diagnosis --usage lp --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --output_dir ./log \
#     --model CLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 \
#     --save_pred

# CUDA_VISIBLE_DEVICES=6 python run_train.py \
#     --task diagnosis --usage lp --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --output_dir ./log \
#     --model BLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 &

# CUDA_VISIBLE_DEVICES=4 python run_train.py \
#     --task diagnosis --usage lp --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# CUDA_VISIBLE_DEVICES=5 python run_train.py \
#     --task diagnosis --usage lp --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# CUDA_VISIBLE_DEVICES=4 python run_train.py \
#     --task diagnosis --usage lp --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# CUDA_VISIBLE_DEVICES=5 python run_train.py \
#     --task diagnosis --usage lp --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --output_dir ./log \
#     --model PLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# CUDA_VISIBLE_DEVICES=6 python run_train.py \
#     --task diagnosis --usage lp --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches\
#     --output_dir ./log \
#     --model SigLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# CUDA_VISIBLE_DEVICES=7 python run_train.py \
#     --task diagnosis --usage lp --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches\
#     --output_dir ./log \
#     --model MedSigLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# CUDA_VISIBLE_DEVICES=7 python run_train.py \
#     --task diagnosis --usage lp --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches\
#     --output_dir ./log \
#     --model PubMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# CUDA_VISIBLE_DEVICES=3 python run_train.py \
#     --task diagnosis --usage lp --dataset Camelyon17 --split train \
#     --image_path ./data/camelyon17_v1.0/patches \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5



# ChestXray (not usig anymore)
# CUDA_VISIBLE_DEVICES=7 python run_train.py \
#     --task diagnosis --usage lp --dataset ChestXray --split train \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --output_dir ./log \
#     --model CLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 &

# CUDA_VISIBLE_DEVICES=6 python run_train.py \
#     --task diagnosis --usage lp --dataset ChestXray --split train \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --output_dir ./log \
#     --model BLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 &

# CUDA_VISIBLE_DEVICES=5 python run_train.py \
#     --task diagnosis --usage lp --dataset ChestXray --split train \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 &

# CUDA_VISIBLE_DEVICES=4 python run_train.py \
#     --task diagnosis --usage lp --dataset ChestXray --split train \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 &

# CUDA_VISIBLE_DEVICES=3 python run_train.py \
#     --task diagnosis --usage lp --dataset ChestXray --split train \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 &


# CUDA_VISIBLE_DEVICES=2 python run_train.py \
#     --task diagnosis --usage lp --dataset ChestXray --split train \
#     --image_path /mnt/sdc/rjin02/DataSets/chest_xray \
#     --output_dir ./log \
#     --model PMCCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5 &




# Train DermaMNIST
# CUDA_VISIBLE_DEVICES=3 python run_train.py \
#     --task diagnosis --usage lp --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model CLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5



# CUDA_VISIBLE_DEVICES=0 python run_train.py \
#     --task diagnosis --usage lp --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5


# CUDA_VISIBLE_DEVICES=5 python run_train.py \
#     --task diagnosis --usage lp --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BLIP2-2.7b --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5


# CUDA_VISIBLE_DEVICES=2 python run_train.py \
#     --task diagnosis --usage lp --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model BioMedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5


# CUDA_VISIBLE_DEVICES=1 python run_train.py \
#     --task diagnosis --usage lp --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model XrayGPT --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5


# CUDA_VISIBLE_DEVICES=6 python run_train.py \
#     --task diagnosis --usage lp --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model MedCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5

# CUDA_VISIBLE_DEVICES=1 python run_train.py \
#     --task diagnosis --usage lp --dataset DermaMNIST --split train \
#     --image_path ./data \
#     --output_dir ./log \
#     --model PMCCLIP --model_path not_given \
#     --cache_dir ./cache \
#     --num_train_epochs 10 \
#     --learning_rate 5e-5
#!/bin/bash
#SBATCH --job-name=squad-erm
#SBATCH --partition=gpu-a40
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bparan@uw.edu

source ~/.bashrc
conda activate work

python run_qa.py \
  --model_name_or_path roberta-base \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --save_total_limit 2 \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/squad_erm_bz32_epoch2 \
  --overwrite_cache --overwrite_output_dir \
  --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/QA/cache_ \

#!/bin/bash
#SBATCH --job-name=sst2
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bparan@uw.edu

source ~/.bashrc
conda activate work
#python run_glue_cartography.py   --model_name_or_path roberta-base --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/train.json --validation_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/validation.json   --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/test.json   --do_train --do_eval --do_predict  --max_seq_length 512   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 8   --output_dir /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/amazon_erm --overwrite_cache --overwrite_output_dir --cache /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/cache_

python run_glue_cartography.py \
    --model_name_or_path roberta-base \
    --custom_task_name sst2 \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/SST2/sst_train.json \
    --validation_file  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/SST2/sst_validation.json  \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/SST2/sst_validation.json \
    --save_total_limit 2 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --output_dir  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/sst2_erm_roberta_bz32_epoch5 \
    --overwrite_cache --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/SST2/cache_ \

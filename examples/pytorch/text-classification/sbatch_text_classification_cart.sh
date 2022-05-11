#!/bin/bash
#SBATCH --job-name=mnli-gdro-em
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
conda activate hgf
#python run_glue_cartography.py   --model_name_or_path roberta-base --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/train.json --validation_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/validation.json   --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/test.json   --do_train --do_eval --do_predict  --max_seq_length 512   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 8   --output_dir /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/amazon_erm --overwrite_cache --overwrite_output_dir --cache /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/cache_

python run_glue_cartography.py \
    --model_name_or_path roberta-base \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/tagged/train_resplit_cartography.json \
    --validation_file  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/tagged/dev_resplit_cartography.json \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/tagged/dev_resplit_cartography.json \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_gdro_eg_roberta_bz32_epoch3_correctness_with_labels \
    --overwrite_cache --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/cache_ \
    --is_robust \
    --reweight_groups \
    --logging_steps 50 \

#--automatic_adjustment \

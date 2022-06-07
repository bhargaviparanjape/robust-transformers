#!/bin/bash
#SBATCH --job-name=mnli-gdro-em
#SBATCH --partition=gpu-a40
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bparan@uw.edu

source ~/.bashrc
conda activate work
#python run_glue_cartography.py   --model_name_or_path roberta-base --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/train.json --validation_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/validation.json   --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/test.json   --do_train --do_eval --do_predict  --max_seq_length 512   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 8   --output_dir /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/amazon_erm --overwrite_cache --overwrite_output_dir --cache /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/cache_

set CUDA_LAUNCH_BLOCKING=1
python run_glue_cartography.py \
    --model_name_or_path roberta-base \
    --custom_task_name mnli_resplit \
    --robust_algorithm GCDRO \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/train_resplit.json \
    --validation_file  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/dev_resplit.json \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/dev_resplit.json \
    --save_total_limit 2 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 256 \
    --learning_rate 2e-5 \
    --num_train_epochs 35 \
    --output_dir  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_gcdro_roberta_bz256_epochs35_betacover \
    --overwrite_cache --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/cache_ \
    --is_robust \
    --logging_steps 50 \
    --metric_for_best_model eval_worst_accuracy \
    --gamma 0.5 \
    --alpha 0.2 \
    --beta 0.5 \
    --beta_ema 0.5 \
    --do_instance_reweight \

#--do_instance_reweight \

#--dro-alpha 0.2 --beta-cover-instances 0.5 --ema 0.5 --beta-ema 0.5 \ 
#--do_instance_reweight \
#--reweight_groups \
#--automatic_adjustment \

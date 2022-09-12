#!/bin/bash
#SBATCH --job-name=qqp
#SBATCH --partition=gpu-rtx6k
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
#python run_glue_cartography.py   --model_name_or_path roberta-base --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/train.json --validation_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/validation.json   --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/test.json   --do_train --do_eval --do_predict  --max_seq_length 512   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 8   --output_dir /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/amazon_erm --overwrite_cache --overwrite_output_dir --cache /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/amazonreviews/experimental/hgf/cache_

python -m torch.distributed.launch --nproc_per_node=1 run_glue_cartography.py \
    --model_name_or_path roberta-base \
    --custom_task_name qqp \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/QQP/automatic_slicing/qqp_train_slicing_error_aware_6slices.json \
    --validation_file  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/QQP/automatic_slicing/qqp_dev_slicing_error_aware_6slices.json  \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/QQP/automatic_slicing/qqp_dev_slicing_error_aware_6slices.json \
    --save_total_limit 2 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 256 \
    --per_device_train_batch_size 144 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --output_dir  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/qqp_domino_roberta_bz144_epoch10 \
    --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/QQP/cache_ \
    --metric_for_best_model eval_worst_accuracy \
    --save_strategy epoch \
    --is_robust \
    --robust_algorithm GCDRO \
    --gamma 0.5 \
    --alpha 0.2 \
    --beta 0.5 \
    --report_to none \
    --do_instance_reweight \


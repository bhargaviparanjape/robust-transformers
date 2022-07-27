#!/bin/bash
#SBATCH --job-name=mnli-erm
#SBATCH --partition=gpu-a40
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

#python -m torch.distributed.launch --nproc_per_node=1 learn_groups.py \
python learn_groups.py \
    --model_name_or_path /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz256_epochs30/checkpoint-4030 \
    --custom_task_name mnli_resplit \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/train_resplit.json \
    --validation_file  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/dev_resplit.json \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/test_resplit.json \
    --save_total_limit 30 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size  144 \
    --per_device_eval_batch_size 256 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --output_dir  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_adversarial_bz144_epochs5 \
    --overwrite_cache --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/cache_ \
    --logging_steps 50 \
    --metric_for_best_model eval_worst_accuracy \
    --is_robust \
    --robust_algorithm GCDRO \
    --adversary_model_name_or_path /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_domino_9_slices_mlp/pytorch_model.bin \
    --train_feature_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz32_epochs3/clustering/error_aware_output_9_slices.pkl \
    --validation_feature_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz32_epochs3/clustering/dev_output_9_slices.pkl \
    --gamma 0.5 \
    --alpha 0.2 \
    --beta 0.5 \

#--save_strategy epoch \

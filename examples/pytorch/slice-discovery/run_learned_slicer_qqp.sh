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

python -m torch.distributed.launch --nproc_per_node=2 learn_groups.py \
    --model_name_or_path /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_deberta_bz128/checkpoint-4000 \
    --custom_task_name mnli_resplit \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/train_resplit_with_features.json \
    --validation_file  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/dev_resplit_with_features.json \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/dev_resplit_with_features.json \
    --save_total_limit 20 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size  72 \
    --per_device_eval_batch_size 96 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --output_dir  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_deberta_adversarial_bz128_epochs20_twostaged_9slices \
    --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/cache_ \
    --logging_steps 50 \
    --metric_for_best_model eval_worst_accuracy \
    --is_robust \
    --robust_algorithm GCDRO \
    --train_feature_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz32_epochs3/clustering/error_aware_output_9_slices.pkl \
    --validation_feature_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz32_epochs3/clustering/dev_output_9_slices.pkl \
    --gamma 0.5 \
    --alpha 0.2 \
    --beta 0.5 \
    --adversary_model_name_or_path /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_domino_9_slices_mlp/pytorch_model.bin \
    --save_strategy epoch \
    --n_slices 9 \
    --adversary_warmup 5 \
    --report_to none \

#--adversary_model_name_or_path /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_domino_9_slices_mlp/pytorch_model.bin \

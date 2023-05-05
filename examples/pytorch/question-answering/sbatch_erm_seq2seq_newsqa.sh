#!/bin/bash
#SBATCH --job-name=newsqa-flan-questiononly
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gpus=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bparan@uw.edu

source ~/.bashrc
conda activate work

export TRANSFORMERS_CACHE=/gscratch/zlab/bparan/projects/transformers_cache

python -m torch.distributed.launch --nproc_per_node=8 --master_port=22686  run_seq2seq_qa.py \
    --model_name_or_path google/flan-t5-base \
    --dataset_name newsqa_custom \
    --dataset_config_name newsqa_custom \
    --data_dir /mmfs1/gscratch/zlab/bparan/projects/causality/data/newsqa/newsqa \
    --do_train \
    --version_2_with_negative \
    --learning_rate 3e-5 \
	--overwrite_output_dir \
	--metric_for_best_model eval_accuracy \
    --num_train_epochs 4 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir /gscratch/zlab/bparan/projects/counterfactuals/models/qa_models/t5-flan_base_newsqa_questiononly \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 8 \
    --save_steps 5000 \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/QA/cache_ \
    --report_to none \
    --overwrite_cache \
    --question_only \

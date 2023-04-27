#!/bin/bash
#SBATCH --job-name=t5-squad-passageonly
#SBATCH --partition=gpu-a40
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gpus=3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bparan@uw.edu

source ~/.bashrc
conda activate work

export TRANSFORMERS_CACHE=/gscratch/zlab/bparan/projects/transformers_cache

python -m torch.distributed.launch --nproc_per_node=3 --master_port=44366  run_seq2seq_qa.py \
    --model_name_or_path google/t5-v1_1-large \
    --dataset_name squad_v2 \
    --do_train \
    --version_2_with_negative \
    --learning_rate 3e-5 \
	--overwrite_output_dir \
	--metric_for_best_model eval_accuracy \
    --num_train_epochs 5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /gscratch/zlab/bparan/projects/counterfactuals/src/t5-1_1_large_squad2.0_passageonly \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 8 \
    --save_steps 5000 \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/QA/cache_ \
    --report_to none \
    --overwrite_cache \
    --passage_only \

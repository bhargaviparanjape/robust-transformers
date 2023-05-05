#!/bin/bash
#SBATCH --job-name=squad-deberta
#SBATCH --partition=gpu-a40
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gpus=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bparan@uw.edu

source ~/.bashrc
conda activate work

export TRANSFORMERS_CACHE=/gscratch/zlab/bparan/projects/transformers_cache


MODEL_NAME=$3
PARTIAL=$1
THRESHOLD=$2
SEED=$4
python -m torch.distributed.launch --nproc_per_node=1 --master_port=23466  run_qa_beam_search.py \
    --model_name_or_path /gscratch/zlab/bparan/projects/counterfactuals/models/qa_models/${MODEL_NAME} \
    --dataset_name newsqa_custom \
    --dataset_config newsqa_custom \
    --data_dir /mmfs1/gscratch/zlab/bparan/projects/causality/data/newsqa/newsqa \
    --do_eval \
    --version_2_with_negative \
    --learning_rate 3e-5 \
	--overwrite_output_dir \
	--metric_for_best_model eval_accuracy \
    --num_train_epochs 5 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_device_eval_batch_size 64 \
    --per_device_train_batch_size 16 \
    --save_steps 5000 \
    --output_dir /gscratch/zlab/bparan/projects/counterfactuals/models/qa_models/roberta_tmp \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/QA/cache_ \
    --report_to none \
    --overwrite_cache \
    --no_answer_threshold ${THRESHOLD} \
    --partial_inputs ${PARTIAL} \
    --partial_inputs_seed ${SEED} \

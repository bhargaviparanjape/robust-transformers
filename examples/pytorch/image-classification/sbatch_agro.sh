#!/bin/bash
#SBATCH --job-name=celebA-kfold4
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

export TRANSFORMERS_CACHE=/gscratch/zlab/bparan/projects/transformers_cache
data_dir=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/WILDS/data/celebA_v1.0
model_dir=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models
#rm -rf ${data_dir}/image_folder
#rm -rf ${data_dir}/_cache/image_folder
python -m torch.distributed.launch --nproc_per_node=1 learn_grouper.py \
    --model_name_or_path ${model_dir}/celebA_vit_erm/checkpoint-4324 \
    --task_name celebA \
    --dataset_name ${data_dir} \
    --train_dir train/*/*.jpg \
    --validation_dir validation/*/*.jpg \
    --output_dir ${model_dir}/celebA_vit_domino_epochs15_bz128_numgroups_8 \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_predict \
    --learning_rate 2e-5 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --logging_strategy steps \
    --logging_steps 50 \
    --evaluation_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 5 \
    --seed 1337 \
	--report_to none \
	--save_strategy epoch \
	--overwrite_output_dir \
	--train_metadata_file ${data_dir}/train_metadata.json \
	--validation_metadata_file ${data_dir}/validation_metadata.json \
	--train_feature_file ${data_dir}/train_features.json \
	--validation_feature_file ${data_dir}/validation_features.json \
	--cache_dir ${data_dir}/_cache \
	--adversary_model_name_or_path ${model_dir}/celeba_domino_8_slices_mlp/pytorch_model.bin \
    --logging_steps 50 \
	--metric_for_best_model eval_worst_accuracy \
	--is_robust \
	--robust_algorithm GCDRO \
	--report_to none \
	--gamma 0.5 \
	--alpha 0.2 \
	--beta 0.5 \
	--n_slices 8 \
	--n_features 771 \
	--adversary_warmup -2 \

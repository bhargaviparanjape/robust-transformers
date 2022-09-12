#!/bin/bash
#SBATCH --job-name=celeba-classification
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

export TRANSFORMERS_CACHE=/gscratch/zlab/bparan/projects/transformers_cache
data_dir=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/WILDS/data/celebA_v1.0
model_dir=/gscratch/zlab/bparan/projects/counterfactuals/models/celebA_vit_agro_epochs15_warmup5_bz128_numgroups_8/best_checkpoint
#rm -rf ${data_dir}/image_folder
#rm -rf ${data_dir}/_cache/image_folder
python run_image_classification_dro.py \
    --task_name celeba \
    --dataset_name ${data_dir} \
    --model_name_or_path ${model_dir} \
    --train_dir train/*/*.jpg \
    --validation_dir validation/*/*.jpg \
    --output_dir ${model_dir} \
    --remove_unused_columns False \
    --do_eval \
    --do_predict \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 256 \
    --logging_strategy steps \
    --logging_steps 10 \
    --load_best_model_at_end True \
    --save_total_limit 5 \
    --seed 1337 \
	--report_to none \
	--overwrite_output_dir \
    --train_metadata_file ${data_dir}/train_metadata.json \
    --validation_metadata_file ${data_dir}/validation_metadata.json \
	--save_strategy epoch \
	--evaluation_strategy epoch \
	--cache_dir /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/WILDS/data/celebA_v1.0/_cache \
	--is_robust \

#!/bin/bash
#SBATCH --job-name=waterbirds-kfold2
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

model_dir=/gscratch/zlab/bparan/projects/counterfactuals/models
data=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/WILDS/data/waterbirds_v1.0
#rm -rf ${data}/image_folder
#rm -rf ${data}/_cache/image_folder
python generate_features.py \
    --task_name waterbirds \
    --dataset_name ${data} \
    --train_dir train/*/*.jpg \
    --validation_dir validation/*/*.jpg \
    --output_dir ${model_dir}/waterbirds_vit_erm \
    --model_name_or_path ${model_dir}/waterbirds_vit_erm \
    --cluster_dev_features \
    --kfold_model_path_prefix /gscratch/zlab/bparan/projects/counterfactuals/models/waterbirds_vit_kfold \
    --n_slices 6 \
    --init_type confusion \
    --n_mixture_components 6 \
    --cluster_assgn_file ${model_dir}/waterbirds_vit_erm/clustering/error_aware_output_6_slices.pkl \
    --output_file train_error_aware_6slices_metadata.json \
    --per_device_train_batch_size 768 \
    --per_device_eval_batch_size 768 \
    --overwrite_output_dir \
    --cache_dir ${data}/_cache \
    --report_to none \
    --train_metadata_file ${data}/train_metadata.json \
    --validation_metadata_file ${data}/validation_metadata.json \
    --remove_unused_columns False \

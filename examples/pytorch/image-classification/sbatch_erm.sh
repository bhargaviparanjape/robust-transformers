#!/bin/bash
#SBATCH --job-name=water-kfold4
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
data_dir=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/WILDS/data/waterbirds_v1.0
#rm -rf ${data_dir}/image_folder
#rm -rf ${data_dir}/_cache/image_folder
python -m torch.distributed.launch --nproc_per_node=1 run_image_classification.py \
    --model_name_or_path microsoft/resnet-50 \
    --task_name waterbirds \
    --dataset_name ${data_dir} \
    --train_dir train/*/*.jpg \
    --validation_dir validation/*/*.jpg \
    --output_dir /gscratch/zlab/bparan/projects/counterfactuals/models/waterbirds_resnet_erm \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_predict \
    --learning_rate 1e-5 \
    --weight_decay 1 \
    --num_train_epochs 300 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy epoch \
    --save_total_limit 5 \
    --metric_for_best_model eval_accuracy \
    --seed 1337 \
	--report_to none \
	--save_strategy epoch \
	--overwrite_output_dir \
	--cache_dir /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/WILDS/data/waterbirds_v1.0/_cache \

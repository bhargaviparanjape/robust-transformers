model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/qqp_erm_roberta_bz32_epoch3
train_file=qqp_train.json
eval_file=qqp_validation.json
data=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/QQP
python error_aware_slice_discovery.py \
    --kfold_model_path_prefix /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/qqp_erm_roberta_bz32_epoch3_kfold \
    --kfold_data_path_prefix ${data}/kfold/train_fold_ \
    --model_name_or_path ${model}/best_checkpoint \
    --pretrained_model_name_or_path roberta-base \
    --custom_task_name qqp \
    --assign_dev_groups \
    --n_slices 6 \
    --init_type confusion \
    --n_mixture_components 6 \
    --train_file ${data}/${train_file} \
    --validation_file ${data}/${eval_file} \
    --test_file ${data}/${eval_file} \
    --cluster_assgn_file ${model}/clustering/dev_output_6_slices.pkl \
    --output_file ${data}/automatic_slicing/qqp_dev_slicing_error_aware_6slices.json \
    --max_seq_length 128 \
    --per_device_train_batch_size 768 \
    --per_device_eval_batch_size 768 \
    --output_dir ${model} \
    --overwrite_output_dir \
    --overwrite_cache \
    --cache_dir ${data}/cache_ \

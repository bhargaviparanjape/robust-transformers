data=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/WILDS/data/civilcomments_v1.0

model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/civilcomments_erm_roberta_bz32_epoch3
train_file=train_v2.json
eval_file=dev_v2.json
python error_aware_slice_discovery.py \
    --kfold_model_path_prefix /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/civilcomments_erm_roberta_bz32_epoch3_kfold \
    --kfold_data_path_prefix ${data}/kfold/train_fold_ \
    --model_name_or_path ${model} \
    --pretrained_model_name_or_path roberta-base \
    --custom_task_name wilds_civil_comments \
    --create_features \
    --cluster_assgn_file ${model}/clustering/error_aware_output_12_slices.pkl \
    --output_file ${data}/automatic_slicing/train_resplit_slicing_error_aware_12slices.json \
    --n_slices 12 \
    --init_type confusion \
    --n_mixture_components 12 \
    --train_file ${data}/${train_file} \
    --validation_file ${data}/${eval_file} \
    --test_file ${data}/${eval_file} \
    --max_seq_length 128 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 512 \
    --output_dir ${model} \
    --overwrite_output_dir \
    --overwrite_cache \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/WILDS/cache_

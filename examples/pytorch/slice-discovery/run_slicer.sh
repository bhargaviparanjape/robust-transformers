model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/civilcomments_erm_roberta_bz32_epoch3
train_file=WILDS/data/civilcomments_v1.0/train_v2.json
eval_file=WILDS/data/civilcomments_v1.0/dev_v2.json
python generate_features.py \
    --model_name_or_path ${model} \
    --custom_task_name wilds_civil_comments \
    --cluster_dev_features \
    --n_slices 24 \
    --init_type confusion \
    --n_mixture_components 24 \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${train_file} \
    --validation_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${eval_file} \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${eval_file} \
    --max_seq_length 256 \
    --per_device_train_batch_size 768 \
    --per_device_eval_batch_size 768 \
    --output_dir ${model} \
    --overwrite_output_dir \
    --overwrite_cache \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/WILDS/cache_ \

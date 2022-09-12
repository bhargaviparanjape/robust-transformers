model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz32_epochs3
train_file=train_resplit.json
eval_file=dev_resplit.json
data=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI
python generate_features.py \
    --model_name_or_path ${model} \
    --custom_task_name mnli_resplit \
    --cluster_dev_features \
    --n_slices 500 \
    --init_type confusion \
    --n_mixture_components 500 \
    --train_file ${data}/${train_file} \
    --validation_file ${data}/${eval_file} \
    --test_file ${data}/${eval_file} \
    --cluster_assgn_file ${model}/clustering/dev_output_500_slices.pkl \
    --output_file ${data}/automatic_slicing/dev_resplit_slicing_error_aware_500slices.json \
    --max_seq_length 128 \
    --per_device_train_batch_size 768 \
    --per_device_eval_batch_size 768 \
    --output_dir ${model} \
    --overwrite_output_dir \
    --overwrite_cache \
    --cache_dir ${data}/cache_ \

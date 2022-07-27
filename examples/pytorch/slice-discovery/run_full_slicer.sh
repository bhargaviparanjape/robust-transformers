data=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI
model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz32_epochs3
python error_aware_slice_discovery.py \
    --kfold_model_path_prefix /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz32_epoch3_kfold \
    --kfold_data_path_prefix ${data}/kfold/train_fold_ \
    --model_name_or_path ${model} \
    --pretrained_model_name_or_path roberta-base \
    --custom_task_name mnli_resplit \
    --assign_train_groups \
    --cluster_assgn_file ${model}/clustering/error_aware_output_24_slices.pkl \
    --output_file ${data}/automatic_slicing/train_resplit_slicing_error_aware_24slices.json \
    --n_slices 24 \
    --init_type confusion \
    --n_mixture_components 24 \
    --train_file ${data}/train_resplit.json \
    --validation_file ${data}/dev_resplit.json \
    --test_file ${data}/test_resplit.json \
    --max_seq_length 128 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --output_dir ${model} \
    --overwrite_output_dir \
    --overwrite_cache \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/cache_ \

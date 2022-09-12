model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/sst2_erm_roberta_bz32_epoch5
train_file=sst_train.json
eval_file=sst_validation.json
data=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/SST2
python error_aware_slice_discovery.py \
    --kfold_model_path_prefix /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/sst2_erm_roberta_bz32_epoch3_kfold \
    --kfold_data_path_prefix ${data}/kfold/train_fold_ \
    --model_name_or_path ${model} \
    --pretrained_model_name_or_path roberta-base \
    --custom_task_name sst2 \
    --assign_dev_groups \
    --n_slices 6 \
    --init_type confusion \
    --n_mixture_components 6 \
    --train_file ${data}/${train_file} \
    --validation_file ${data}/${eval_file} \
    --test_file ${data}/${eval_file} \
    --cluster_assgn_file ${model}/clustering/dev_output_6_slices.pkl \
    --output_file ${data}/automatic_slicing/sst2_dev_slicing_error_aware_6slices.json \
    --max_seq_length 128 \
    --per_device_train_batch_size 768 \
    --per_device_eval_batch_size 768 \
    --output_dir ${model} \
    --overwrite_output_dir \
    --overwrite_cache \
    --cache_dir ${data}/cache_ \

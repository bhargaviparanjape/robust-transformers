model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz32_epochs3
python generate_pretrained_features.py \
    --model_name_or_path ${model} \
    --pretrained_model_name_or_path roberta-base \
    --custom_task_name mnli_resplit \
    --cluster_dev_features \
    --n_slices 500 \
    --init_type confusion \
    --n_mixture_components 500 \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/train_resplit.json \
    --validation_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/dev_resplit.json \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/test_resplit.json \
    --max_seq_length 128 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --output_dir ${model} \
    --overwrite_output_dir \
    --overwrite_cache \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/cache_ \

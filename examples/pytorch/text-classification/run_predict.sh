#model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/automatic_clustering/mnli_erm_rw_roberta_bz32_epoch3_slices24_devbased
model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_gcdro_roberta_bz256_epochs35_betacover_domino12slices_erroraware
train_dataset=NLI/MNLI/dev_resplit.json
dataset=NLI/MNLI/dev_lex.json
python run_glue_cartography.py \
    --model_name_or_path ${model} \
    --custom_task_name mnli_resplit \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${train_dataset} \
    --validation_file  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${dataset} \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${dataset}  \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --learning_rate 2e-5 \
    --num_train_epochs 8 \
    --output_dir  ${model} \
    --overwrite_cache --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/cache_ \
#    --transform_labels \

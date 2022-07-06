#model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/automatic_clustering/mnli_erm_rw_roberta_bz32_epoch3_slices24_devbased
model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/civilcomments_gcdro_roberta_bz256_epoch30_error_aware_domino12slices_evaldomino12/best_checkpoint
train_dataset=WILDS/data/civilcomments_v1.0/train_v2.json
dataset=WILDS/data/civilcomments_v1.0/dev_v2.json
python run_glue_cartography.py \
    --model_name_or_path ${model} \
    --custom_task_name wilds_civil_comments \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${train_dataset} \
    --validation_file  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${dataset} \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${dataset}  \
    --do_eval \
    --do_predict \
    --max_seq_length 256 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --learning_rate 2e-5 \
    --num_train_epochs 8 \
    --output_dir  ${model} \
    --overwrite_cache --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/WILDS/cache_ \
#    --transform_labels \

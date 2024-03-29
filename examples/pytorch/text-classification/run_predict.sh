model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz256_epoch10
#model=microsoft/deberta-base-mnli
train_dataset=NLI/nli-debiasing-datasets/robust_nli.json
dataset=NLI/nli-debiasing-datasets/robust_nli.json
python run_glue_cartography.py \
    --model_name_or_path ${model} \
    --custom_task_name mnli_resplit \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${train_dataset} \
    --validation_file  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${dataset} \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/${dataset}  \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --output_dir  ${model} \
    --overwrite_cache --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/cache_ \
    --report_to none \
#    --transform_labels \

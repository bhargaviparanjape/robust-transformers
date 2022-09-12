model=$1

# original Dev set
rm ${model}/all_sst2_results.json
for dataset in imdb_cad imdb_contrast imdb semeval_twoclass_test senti140_twoclass_test yelp_polarity
do
    train_dataset=SST2/ood/${dataset}.json
    dataset=SST2/ood/${dataset}.json
    python run_glue_cartography.py \
        --model_name_or_path ${model} \
        --custom_task_name sst2 \
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
        --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/SST2/cache_
    cat ${model}/eval_results.json >> ${model}/all_sst2_results.json
done

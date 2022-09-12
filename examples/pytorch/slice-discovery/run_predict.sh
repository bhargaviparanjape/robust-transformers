model=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_deberta_adversarial_bz128_epochs20_twostaged_9slices
export CUDA_VISIBLE_DEVICES=0
python learn_groups.py \
    --model_name_or_path ${model}/best_checkpoint \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/dev_resplit_with_features.json \
    --validation_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/dev_resplit_with_features.json \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/dev_resplit_with_features.json \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_eval_batch_size 128 \
    --output_dir /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_adversarial_bz144_epochs5 \
    --output_dir ${model} \
    --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/cache_ \
    --train_feature_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz32_epochs3/clustering/error_aware_output_9_slices.pkl \
    --validation_feature_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz32_epochs3/clustering/dev_output_9_slices.pkl \
    --report_to none \
    --is_robust \
    #--select_predicted_worst_group \
    #--overwrite_cache \
    #--transform_labels \

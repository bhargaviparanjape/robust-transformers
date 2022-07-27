model=$1

# original Dev set
train_dataset=NLI/MNLI/dev_resplit.json
dataset=NLI/MNLI/dev_resplit.json
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
cat ${model}/eval_results.json >> ${model}/combined_results.json

train_dataset=NLI/MNLI/dev_resplit.json
dataset=NLI/MNLI/dev_worst.json
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
cat ${model}/eval_results.json >> ${model}/combined_results.json

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
cat ${model}/eval_results.json >> ${model}/combined_results.json

train_dataset=NLI/MNLI/eval_sets/snli_validation.json
dataset=NLI/MNLI/eval_sets/snli_validation.json
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
cat ${model}/eval_results.json >> ${model}/combined_results.json

train_dataset=NLI/MNLI/eval_sets/anli_validation_r2.json
dataset=NLI/MNLI/eval_sets/anli_validation_r2.json
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
cat ${model}/eval_results.json >> ${model}/combined_results.json

train_dataset=NLI/wanli/wanli/validation_reformatted.json
dataset=NLI/wanli/wanli/validation_reformatted.json
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
cat ${model}/eval_results.json >> ${model}/combined_results.json

train_dataset=NLI/MNLI/eval_sets/snli_validation.json
dataset=NLI/MNLI/eval_sets/hans_validation.json
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
    --transform_labels true \
cat ${model}/eval_results.json >> ${model}/combined_results.json

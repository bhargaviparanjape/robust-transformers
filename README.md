# AGRO: Adversarial discovery of error-prone Groups for Robust Optimization 

## Requirements

Python>=3.8 and install transformers as editable.
```
pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
git clone https://github.com/bhargaviparanjape/robust-transformers.git
cd transformers
pip install -e .
```

## Instructions


To prepare features for grouper model:

```
python run_glue_cartography.py \
    --model_name_or_path roberta-base \
    --custom_task_name mnli_resplit \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/kfold/train_fold_4.json \
    --validation_file  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/kfold/dev_fold_4.json  \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/kfold/dev_fold_4.json \
    --save_total_limit 2 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_erm_roberta_bz32_epoch3_kfold4 \
    --overwrite_cache --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/cache_ \
```

For text classification (GCDRO):

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=25666 run_glue_cartography.py \
    --model_name_or_path microsoft/deberta-base \
    --custom_task_name mnli_resplit \
    --train_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/train_george.json \
    --validation_file  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/dev_george.json \
    --test_file /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/dev_george.json \
    --save_total_limit 15 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 96 \
    --per_device_eval_batch_size 256 \
    --learning_rate 1e-5 \
    --weight_decay 0.5 \
    --num_train_epochs 20 \
    --output_dir  /mmfs1/gscratch/zlab/bparan/projects/counterfactuals/models/mnli_gcdro_deberta_bz96_george_ep20 \
    --overwrite_cache --overwrite_output_dir \
    --cache_dir /gscratch/zlab/bparan/projects/counterfactuals/data/NLI/MNLI/cache_ \
    --logging_steps 50 \
    --metric_for_best_model eval_worst_accuracy \
    --save_strategy epoch \
    --is_robust \
    --robust_algorithm GCDRO \
    --gamma 0.5 \
    --alpha 0.2 \
    --beta 0.5 \
    --report_to none \
```

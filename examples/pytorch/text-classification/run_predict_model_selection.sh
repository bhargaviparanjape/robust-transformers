data=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data
train_dataset=NLI/MNLI/dev_resplit.json
dataset=NLI/MNLI/dev_worst.json


for dir in ${1}/*/
do
    if [[ ${dir} == *"checkpoint-"* ]];
    then
        echo ${dir}
        python run_glue_cartography.py --model_name_or_path ${dir} --custom_task_name mnli_resplit --train_file ${data}/${train_dataset} --validation_file  ${data}/${dataset} --test_file ${data}/${dataset}  --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --learning_rate 2e-5 --num_train_epochs 8 --output_dir  ${dir} --overwrite_cache --overwrite_output_dir --cache_dir ${data}/NLI/MNLI/cache_
    fi
done



output_file=${1}/dev_worst_validation.txt
rm ${output_file}
for dir in ${1}/*/
do
    if [[ ${dir} == *"checkpoint-"* ]];
    then
        echo ${dir} >> ${output_file}
        grep -o '"eval_accuracy": [0-9\.]*' ${dir}/eval_results.json | grep -o '[0-9\.]*' >> ${output_file}
    fi
done

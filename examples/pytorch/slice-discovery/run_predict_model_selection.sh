data=/mmfs1/gscratch/zlab/bparan/projects/counterfactuals/data
train_dataset=NLI/MNLI/dev_resplit_with_features.json
dataset=NLI/MNLI/dev_resplit_with_features.json

for dir in ${1}/*/
do
    if [[ ${dir} == *"checkpoint-"* ]];
    then
        echo ${dir}
        export CUDA_VISIBLE_DEVICES=0
        python learn_groups.py --model_name_or_path ${dir} --train_file ${data}/${train_dataset} --validation_file  ${data}/${dataset} --test_file ${data}/${dataset}  --do_eval --do_predict --max_seq_length 128 --per_device_eval_batch_size 256  --output_dir  ${dir} --overwrite_output_dir --cache_dir ${data}/NLI/MNLI/cache_  --is_robust --select_predicted_worst_group --report_to none
    fi
done



output_file=${1}/dev_megagroup_validation.txt
rm ${output_file}
for dir in ${1}/*/
do
    if [[ ${dir} == *"checkpoint-"* ]];
    then
        echo ${dir} >> ${output_file}
        cat ${dir}/eval_results.json >> ${output_file}
    fi
done

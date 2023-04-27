: '
./predict.sh same_title -8.612630844 deberta_large_finetuned_squad2.0/checkpoint-75000 2>&1 | tee -a output.log
./predict.sh no -8.612630844 deberta_large_finetuned_squad2.0/checkpoint-75000 2>&1 | tee -a output.log
./predict.sh previous -8.612630844 deberta_large_finetuned_squad2.0/checkpoint-75000 2>&1 | tee -a output.log
./predict.sh random -8.612630844 deberta_large_finetuned_squad2.0/checkpoint-75000 2>&1 | tee -a output.log

./predict.sh same_title -6.980183125 deberta_base_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh no -6.980183125 deberta_base_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh previous -6.980183125 deberta_base_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh random -6.980183125 deberta_base_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log


./predict.sh same_title -10.193 roberta_base_finetuned_squad2.0/checkpoint-10000 2>&1 | tee -a output.log
./predict.sh no -10.193 roberta_base_finetuned_squad2.0/checkpoint-10000 2>&1 | tee -a output.log
./predict.sh previous -10.193 roberta_base_finetuned_squad2.0/checkpoint-10000 2>&1 | tee -a output.log
./predict.sh random -10.193 roberta_base_finetuned_squad2.0/checkpoint-10000 2>&1 | tee -a output.log

./predict.sh same_title -13.7057 roberta_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh no -13.7057 roberta_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh previous -13.7057 roberta_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh random -13.7057 roberta_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
'

./predict.sh none 1.0 bertbase_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 bertlarge_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 roberta_base_finetuned_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 roberta_large_finetuned_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 debertav3_small_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 debertav3_base_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 debertav3_large_squad2.0 2>&1 | tee -a output_squad.log

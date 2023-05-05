: '
./predict.sh no -8.612630844 deberta_large_finetuned_squad2.0/checkpoint-75000 2>&1 | tee -a output.log
./predict.sh no -8.612630844 deberta_large_finetuned_squad2.0/checkpoint-75000 2>&1 | tee -a output.log
./predict.sh previous -8.612630844 deberta_large_finetuned_squad2.0/checkpoint-75000 2>&1 | tee -a output.log
./predict.sh no -8.612630844 deberta_large_finetuned_squad2.0/checkpoint-75000 2>&1 | tee -a output.log

./predict.sh no -6.980183125 deberta_base_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh no -6.980183125 deberta_base_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh previous -6.980183125 deberta_base_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh no -6.980183125 deberta_base_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log


./predict.sh no -10.193 roberta_base_finetuned_squad2.0/checkpoint-10000 2>&1 | tee -a output.log
./predict.sh no -10.193 roberta_base_finetuned_squad2.0/checkpoint-10000 2>&1 | tee -a output.log
./predict.sh previous -10.193 roberta_base_finetuned_squad2.0/checkpoint-10000 2>&1 | tee -a output.log
./predict.sh no -10.193 roberta_base_finetuned_squad2.0/checkpoint-10000 2>&1 | tee -a output.log

./predict.sh no -13.7057 roberta_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh no -13.7057 roberta_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh previous -13.7057 roberta_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log
./predict.sh no -13.7057 roberta_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output.log


./predict.sh none 1.0 bertbase_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 bertlarge_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 roberta_base_finetuned_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 roberta_large_finetuned_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 debertav3_small_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 debertav3_base_squad2.0 2>&1 | tee -a output_squad.log
./predict.sh none 1.0 debertav3_large_squad2.0 2>&1 | tee -a output_squad.log


./predict_partial.sh none 1.0 bert_base_squad_passageonly 2>&1 | tee -a output_squad_partial_passage.log
./predict_partial.sh none 1.0 bert_base_squad_passageonly 2>&1 | tee -a output_squad_partial_passage.log
./predict_partial.sh none 1.0 bert_large_squad_passageonly 2>&1 | tee -a output_squad_partial_passage.log
./predict_partial.sh none 1.0 roberta_base_squad_passageonly 2>&1 | tee -a output_squad_partial_passage.log
./predict_partial.sh none 1.0 roberta_large_squad_passageonly 2>&1 | tee -a output_squad_partial_passage.log
./predict_partial.sh none 1.0 deberta_base_squad_passageonly 2>&1 | tee -a output_squad_partial_passage.log
./predict_partial.sh none 1.0 deberta_large_squad_passageonly 2>&1 | tee -a output_squad_partial_passage.log
'


for SEED in 433 909 546 862 742
do
    ./predict.sh no -9.0369 bertbase_squad2.0 ${SEED} 2>&1 | tee -a output_squad_no_question_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict.sh no -14.0638 bertlarge_squad2.0 ${SEED} 2>&1 | tee -a output_squad_no_question_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict.sh no -10.1623 roberta_base_finetuned_squad2.0 ${SEED} 2>&1 | tee -a output_squad_no_question_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict.sh no -14.2985 roberta_large_finetuned_squad2.0 ${SEED} 2>&1 | tee -a output_squad_no_question_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict.sh no -9.4429 debertav3_small_squad2.0 ${SEED} 2>&1 | tee -a output_squad_no_question_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict.sh no -9.1722 debertav3_base_squad2.0 ${SEED} 2>&1 | tee -a output_squad_no_question_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict.sh no -14.0009 debertav3_large_squad2.0 ${SEED} 2>&1 | tee -a output_squad_no_question_cf.log
done

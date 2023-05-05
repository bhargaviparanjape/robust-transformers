: '
./predict_seq2seq.sh same_title 0 t5_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq.sh same_title 0 t5_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq.sh previous 0 t5_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq.sh same_title 0 t5_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log

./predict_seq2seq.sh same_title 0 t5_small_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq.sh same_title 0 t5_small_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq.sh previous 0 t5_small_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq.sh same_title 0 t5_small_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log

./predict_seq2seq.sh same_titlene 1.0 t5_small_finetuned_squad2.0 2>&1 | tee -a output_squad_seq2seq.log
./predict_seq2seq.sh same_titlene 1.0 t5_large_finetuned_squad2.0 2>&1 | tee -a output_squad_seq2seq.log
./predict_seq2seq.sh same_titlene 1.0 t5-v1_1-base_finetuned_squad2.0 2>&1 | tee -a output_squad_seq2seq.log
./predict_seq2seq.sh same_titlene 1.0 t5-v1_1-large_finetuned_squad2.0 2>&1 | tee -a output_squad_seq2seq.log
./predict_seq2seq.sh same_titlene 1.0 t5-flan-base_finetuned_squad2.0 2>&1 | tee -a output_squad_seq2seq.log
./predict_seq2seq.sh same_titlene 1.0 t5-flan-large_finetuned_squad2.0 2>&1 | tee -a output_squad_seq2seq.log


./predict_seq2seq_partial.sh same_titlene 1.0 t5-1_1_base_squad2.0_passageonly 2>&1 | tee -a output_squad_seq2seq_passage.log
./predict_seq2seq_partial.sh same_titlene 1.0 t5-1_1_large_squad2.0_passageonly 2>&1 | tee -a output_squad_seq2seq_passage.log
./predict_seq2seq_partial.sh same_titlene 1.0 t5-flan_base_squad2.0_passageonly 2>&1 | tee -a output_squad_seq2seq_passage.log
./predict_seq2seq_partial.sh same_titlene 1.0 t5-flan_large_squad2.0_passageonly 2>&1 | tee -a output_squad_seq2seq_passage.log
'

for SEED in 433 909 546 862 742
do
    ./predict_seq2seq.sh same_title 0.0 t5_small_finetuned_squad2.0 ${SEED} 2>&1 | tee -a output_squad_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq.sh same_title 0.0 t5_large_finetuned_squad2.0 ${SEED} 2>&1 | tee -a output_squad_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq.sh same_title 0.0 t5-v1_1-base_finetuned_squad2.0 ${SEED} 2>&1 | tee -a output_squad_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq.sh same_title 0.0 t5-v1_1-large_finetuned_squad2.0 ${SEED} 2>&1 | tee -a output_squad_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq.sh same_title 0.0 t5-flan-base_finetuned_squad2.0 ${SEED} 2>&1 | tee -a output_squad_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq.sh same_title 0.0 t5-flan-large_finetuned_squad2.0 ${SEED} 2>&1 | tee -a output_squad_seq2seq_same_title_cf.log
done

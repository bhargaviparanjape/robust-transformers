: '
./predict_seq2seq_duorc.sh same_title 0 t5_large_finetuned_duorc/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq_duorc.sh same_title 0 t5_large_finetuned_duorc/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq_duorc.sh previous 0 t5_large_finetuned_duorc/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq_duorc.sh same_title 0 t5_large_finetuned_duorc/checkpoint-40000 2>&1 | tee -a output1.log

./predict_seq2seq_duorc.sh same_title 0 t5_small_finetuned_duorc/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq_duorc.sh same_title 0 t5_small_finetuned_duorc/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq_duorc.sh previous 0 t5_small_finetuned_duorc/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq_duorc.sh same_title 0 t5_small_finetuned_duorc/checkpoint-40000 2>&1 | tee -a output1.log

./predict_seq2seq_duorc.sh same_titlene 1.0 t5_base_finetuned_duorc 2>&1 | tee -a output_duorc_seq2seq.log
./predict_seq2seq_duorc.sh same_titlene 1.0 t5_large_finetuned_duorc 2>&1 | tee -a output_duorc_seq2seq.log
./predict_seq2seq_duorc.sh same_titlene 1.0 t5-v1_1_base_finetuned_duorc 2>&1 | tee -a output_duorc_seq2seq.log
./predict_seq2seq_duorc.sh same_titlene 1.0 t5-v1_1_large_finetuned_duorc 2>&1 | tee -a output_duorc_seq2seq.log
./predict_seq2seq_duorc.sh same_titlene 1.0 t5-flan_base_finetuned_duorc 2>&1 | tee -a output_duorc_seq2seq.log
./predict_seq2seq_duorc.sh same_titlene 1.0 t5-flan_large_finetuned_duorc 2>&1 | tee -a output_duorc_seq2seq.log


./predict_seq2seq_duorc.sh same_titlene 1.0 t5_1_1_base_duorc_passageonly 2>&1 | tee -a output_duorc_seq2seq_passage.log
./predict_seq2seq_duorc.sh same_titlene 1.0 t5_1_1_large_duorc_passageonly 2>&1 | tee -a output_duorc_seq2seq_passage.log
./predict_seq2seq_duorc.sh same_titlene 1.0 t5_flan_base_duorc_passageonly 2>&1 | tee -a output_duorc_seq2seq_passage.log
./predict_seq2seq_duorc.sh same_titlene 1.0 t5_flan_large_duorc_passageonly 2>&1 | tee -a output_duorc_seq2seq_passage.log
'

for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_duorc.sh same_title 0.0 t5_base_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_duorc.sh same_title 0.0 t5_large_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_duorc.sh same_title 0.0 t5-v1_1_base_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_duorc.sh same_title 0.0 t5-v1_1_large_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_duorc.sh same_title 0.0 t5-flan_base_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_duorc.sh same_title 0.0 t5-flan_large_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_seq2seq_same_title_cf.log
done

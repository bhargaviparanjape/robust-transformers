: '
#./predict_seq2seq_newsqa.sh same_title 0 t5_large_finetuned_newsqa/checkpoint-40000 2>&1 | tee -a output1.log
#./predict_seq2seq_newsqa.sh same_title 0 t5_large_finetuned_newsqa/checkpoint-40000 2>&1 | tee -a output1.log
#./predict_seq2seq_newsqa.sh previous 0 t5_large_finetuned_newsqa/checkpoint-40000 2>&1 | tee -a output1.log
#./predict_seq2seq_newsqa.sh same_title 0 t5_large_finetuned_newsqa/checkpoint-40000 2>&1 | tee -a output1.log

#./predict_seq2seq_newsqa.sh same_title 0 t5_small_finetuned_newsqa/checkpoint-40000 2>&1 | tee -a output1.log
#./predict_seq2seq_newsqa.sh same_title 0 t5_small_finetuned_newsqa/checkpoint-40000 2>&1 | tee -a output1.log
#./predict_seq2seq_newsqa.sh previous 0 t5_small_finetuned_newsqa/checkpoint-40000 2>&1 | tee -a output1.log
#./predict_seq2seq_newsqa.sh same_title 0 t5_small_finetuned_newsqa/checkpoint-40000 2>&1 | tee -a output1.log

./predict_seq2seq_newsqa.sh same_titlene 1.0 t5-base_finetuned_newsqa 2>&1 | tee -a output_newsqa_seq2seq.log
./predict_seq2seq_newsqa.sh same_titlene 1.0 t5-large_finetuned_newsqa 2>&1 | tee -a output_newsqa_seq2seq.log
./predict_seq2seq_newsqa.sh same_titlene 1.0 t5-v1_base_finetuned_newsqa 2>&1 | tee -a output_newsqa_seq2seq.log
./predict_seq2seq_newsqa.sh same_titlene 1.0 t5-v1_large_finetuned_newsqa 2>&1 | tee -a output_newsqa_seq2seq.log
./predict_seq2seq_newsqa.sh same_titlene 1.0 t5-flan-base_finetuned_newsqa 2>&1 | tee -a output_newsqa_seq2seq.log
./predict_seq2seq_newsqa.sh same_titlene 1.0 t5-flan-large_finetuned_newsqa 2>&1 | tee -a output_newsqa_seq2seq.log


./predict_seq2seq_newsqa.sh same_titlene 1.0 t5-v1_1_base_newsqa_passageonly 2>&1 | tee -a output_newsqa_seq2seq_passage.log
./predict_seq2seq_newsqa.sh same_titlene 1.0 t5-v1_1_large_newsqa_passageonly 2>&1 | tee -a output_newsqa_seq2seq_passage.log
./predict_seq2seq_newsqa.sh same_titlene 1.0 t5-flan_base_newsqa_passageonly 2>&1 | tee -a output_newsqa_seq2seq_passage.log
./predict_seq2seq_newsqa.sh same_titlene 1.0 t5-flan_large_newsqa_passageonly 2>&1 | tee -a output_newsqa_seq2seq_passage.log
'

for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_newsqa.sh same_title 0.0 t5-base_finetuned_newsqa ${SEED} 2>&1 | tee -a output_newsqa_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_newsqa.sh same_title 0.0 t5-large_finetuned_newsqa ${SEED} 2>&1 | tee -a output_newsqa_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_newsqa.sh same_title 0.0 t5-v1_base_finetuned_newsqa ${SEED} 2>&1 | tee -a output_newsqa_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_newsqa.sh same_title 0.0 t5-v1_large_finetuned_newsqa ${SEED} 2>&1 | tee -a output_newsqa_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_newsqa.sh same_title 0.0 t5-flan-base_finetuned_newsqa ${SEED} 2>&1 | tee -a output_newsqa_seq2seq_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_seq2seq_newsqa.sh same_title 0.0 t5-flan-large_finetuned_newsqa ${SEED} 2>&1 | tee -a output_newsqa_seq2seq_same_title_cf.log
done

: '
#./predict_newsqa.sh same_titlene 1.0 bert_base_newsqa 2>&1 | tee -a output_newsqa.log
#./predict_newsqa.sh same_titlene 1.0 bert_large_newsqa 2>&1 | tee -a output_newsqa.log
#./predict_newsqa.sh same_titlene 1.0 roberta_base_newsqa 2>&1 | tee -a output_newsqa.log
#./predict_newsqa.sh same_titlene 1.0 roberta_large_newsqa 2>&1 | tee -a output_newsqa.log
#./predict_newsqa.sh same_titlene 1.0 debertav3_small_newsqa 2>&1 | tee -a output_newsqa.log
#./predict_newsqa.sh same_titlene 1.0 debertav3_base_newsqa 2>&1 | tee -a output_newsqa.log
#./predict_newsqa.sh same_titlene 1.0 debertav3_large_newsqa 2>&1 | tee -a output_newsqa.log


./predict_newsqa_partial.sh same_titlene 1.0 bert_base_newsqa_passageonly 2>&1 | tee -a output_newsqa_partial_passage.log
./predict_newsqa_partial.sh same_titlene 1.0 bert_large_newsqa_passageonly 2>&1 | tee -a output_newsqa_partial_passage.log
./predict_newsqa_partial.sh same_titlene 1.0 roberta_base_newsqa_passageonly 2>&1 | tee -a output_newsqa_partial_passage.log
./predict_newsqa_partial.sh same_titlene 1.0 roberta_large_newsqa_passageonly 2>&1 | tee -a output_newsqa_partial_passage.log
./predict_newsqa_partial.sh same_titlene 1.0 deberta_base_newsqa_passageonly 2>&1 | tee -a output_newsqa_partial_passage.log
./predict_newsqa_partial.sh same_titlene 1.0 deberta_large_newsqa_passageonly 2>&1 | tee -a output_newsqa_partial_passage.log
'

for SEED in 433 909 546 862 742
do
    ./predict_newsqa.sh same_title -11.5041 bert_base_newsqa ${SEED} 2>&1 | tee -a output_newsqa_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_newsqa.sh same_title -3.7684 bert_large_newsqa ${SEED} 2>&1 | tee -a output_newsqa_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_newsqa.sh same_title -11.2666 roberta_base_newsqa ${SEED}  2>&1 | tee -a output_newsqa_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_newsqa.sh same_title -8.4968 roberta_large_newsqa ${SEED} 2>&1 | tee -a output_newsqa_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_newsqa.sh same_title -8.4417 debertav3_small_newsqa ${SEED} 2>&1 | tee -a output_newsqa_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_newsqa.sh same_title -8.9874 debertav3_base_newsqa ${SEED} 2>&1 | tee -a output_newsqa_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_newsqa.sh same_title -3.5472 debertav3_large_newsqa ${SEED} 2>&1 | tee -a output_newsqa_same_title_cf.log
done

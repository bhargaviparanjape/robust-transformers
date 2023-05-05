: '
./predict_duorc.sh same_title -2.66858363 deberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_title -2.66858363 deberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh previous -2.66858363 deberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_title -2.66858363 deberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log


./predict_duorc.sh same_title -0.0883718 roberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_title -0.0883718 roberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh previous -0.0883718 roberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_title -0.0883718 roberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log

./predict_duorc.sh same_title -1.0167121 roberta_large_finetuned_duorc/checkpoint-35000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_title -1.0167121 roberta_large_finetuned_duorc/checkpoint-35000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh previous -1.0167121 roberta_large_finetuned_duorc/checkpoint-35000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_title -1.0167121 roberta_large_finetuned_duorc/checkpoint-35000 2>&1 | tee -a output_duorc.log

./predict_duorc.sh same_title -3.2879 deberta_large_finetuned_duorc/checkpoint-70000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_title -3.2879 deberta_large_finetuned_duorc/checkpoint-70000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh previous -3.2879 deberta_large_finetuned_duorc/checkpoint-70000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_title -3.2879 deberta_large_finetuned_duorc/checkpoint-70000 2>&1 | tee -a output_duorc.log


./predict_duorc.sh same_titlene 1.0 bert_base_finetuned_duorc 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_titlene 1.0 bert_large_finetuned_duorc 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_titlene 1.0 roberta_base_finetuned_duorc 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_titlene 1.0 roberta_large_finetuned_duorc 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_titlene 1.0 deberta-v3-small_finetuned_duorc 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_titlene 1.0 deberta-v3-base_finetuned_duorc 2>&1 | tee -a output_duorc.log
./predict_duorc.sh same_titlene 1.0 deberta-v3-large_finetuned_duorc 2>&1 | tee -a output_duorc.log


./predict_duorc_partial.sh same_titlene 1.0 bert_base_finetuned_duorc_passageonly 2>&1 | tee -a output_duorc_partial_passage.log
./predict_duorc_partial.sh same_titlene 1.0 bert_large_finetuned_duorc_passageonly 2>&1 | tee -a output_duorc_partial_passage.log
./predict_duorc_partial.sh same_titlene 1.0 roberta_base_finetuned_duorc_passageonly 2>&1 | tee -a output_duorc_partial_passage.log
./predict_duorc_partial.sh same_titlene 1.0 roberta_large_finetuned_duorc_passageonly 2>&1 | tee -a output_duorc_partial_passage.log
./predict_duorc_partial.sh same_titlene 1.0 deberta_base_finetuned_duorc_passageonly 2>&1 | tee -a output_duorc_partial_passage.log
./predict_duorc_partial.sh same_titlene 1.0 deberta_large_finetuned_duorc_passageonly 2>&1 | tee -a output_duorc_partial_passage.log
'

for SEED in 433 909 546 862 742
do
    ./predict_duorc.sh same_title 9.91 bert_base_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_duorc.sh same_title 12.20 bert_large_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_duorc.sh same_title -4.25 roberta_base_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_duorc.sh same_title -3.38 roberta_large_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_duorc.sh same_title 7.85 deberta-v3-small_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_duorc.sh same_title 10.01 deberta-v3-base_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_same_title_cf.log
done
for SEED in 433 909 546 862 742
do
    ./predict_duorc.sh same_title 14.33 deberta-v3-large_finetuned_duorc ${SEED} 2>&1 | tee -a output_duorc_same_title_cf.log
done

: '
./predict_duorc.sh same_title -2.66858363 deberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh no -2.66858363 deberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh previous -2.66858363 deberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh random -2.66858363 deberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log


./predict_duorc.sh same_title -0.0883718 roberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh no -0.0883718 roberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh previous -0.0883718 roberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh random -0.0883718 roberta_base_finetuned_duorc/checkpoint-15000 2>&1 | tee -a output_duorc.log

./predict_duorc.sh same_title -1.0167121 roberta_large_finetuned_duorc/checkpoint-35000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh no -1.0167121 roberta_large_finetuned_duorc/checkpoint-35000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh previous -1.0167121 roberta_large_finetuned_duorc/checkpoint-35000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh random -1.0167121 roberta_large_finetuned_duorc/checkpoint-35000 2>&1 | tee -a output_duorc.log
'
./predict_duorc.sh same_title -3.2879 deberta_large_finetuned_duorc/checkpoint-70000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh no -3.2879 deberta_large_finetuned_duorc/checkpoint-70000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh previous -3.2879 deberta_large_finetuned_duorc/checkpoint-70000 2>&1 | tee -a output_duorc.log
./predict_duorc.sh random -3.2879 deberta_large_finetuned_duorc/checkpoint-70000 2>&1 | tee -a output_duorc.log

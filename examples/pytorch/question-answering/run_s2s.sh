#./predict_seq2seq.sh same_title 0 t5_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq.sh no 0 t5_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
#./predict_seq2seq.sh previous 0 t5_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
#./predict_seq2seq.sh random 0 t5_large_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log

#./predict_seq2seq.sh same_title 0 t5_small_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
./predict_seq2seq.sh no 0 t5_small_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
#./predict_seq2seq.sh previous 0 t5_small_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log
#./predict_seq2seq.sh random 0 t5_small_finetuned_squad2.0/checkpoint-40000 2>&1 | tee -a output1.log

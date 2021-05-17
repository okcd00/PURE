task_name=msra


python run_entity.py \
    --do_eval --eval_test \
    --max_span_length=16 \
    --context_window 0 \
    --task ${task_name} \
    --data_dir ./data/test_${task_name}/ \
    --model bert-base-chinese \
    --bert_model_dir /data/chend/model_files/chinese_L-12_H-768_A-12/ \
    --output_dir ./output_dir/test_${task_name} 

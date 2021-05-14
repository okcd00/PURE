task_name = resume


python run_entity.py \
    --do_train --do_eval --eval_test \
    --learning_rate=1e-5 \
    --task_learning_rate=5e-4 \
    --train_batch_size=16 \
    --context_window 0 \
    --task ${task_name} \
    --data_dir ./data/${task_name} \
    --model bert-base-uncased \
    --output_dir ./output_dir/${task_name} 

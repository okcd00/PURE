task_name=msra


python run_entity.py \
    --do_train \
    --do_eval \
    --eval_test \
    --inv_test \
    --take_width_feature True \
    --take_name_feature True \
    --take_context_feature False \
    --fusion_method none \
    --learning_rate=1e-5 \
    --task_learning_rate=5e-4 \
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --max_span_length=25 \
    --num_epoch 50 \
    --print_loss_step 500 \
    --context_window 0 \
    --model bert-base-chinese \
    --task ${task_name} \
    --data_dir ./data/${task_name} \
    --output_dir ./output_dir/${task_name} \

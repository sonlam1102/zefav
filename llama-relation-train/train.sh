export EPOCHS=10
export EVAL_STEP=0.1

python run_sft.py \
    --model_name lmsys/vicuna-7b-v1.5 \
    --output_dir ./model/vicuna-7b-v1.5/  \
    --dataset_name few_rel \
    --dataset_config_name default \
    --dataset_split train_wiki \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs ${EPOCHS} \
    --torch_dtype bfloat16 \
    --optim adafactor \
    --learning_rate 1e-5 \
    --report_to tensorboard \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEP \
    --logging_steps $EVAL_STEP \
    --save_steps $EVAL_STEP \
    --low_cpu_mem_usage \
    --use_peft \
    --instruction_format_fn format_few_rel

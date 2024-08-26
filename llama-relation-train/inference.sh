python run_inference.py \
    --model_path ./model/llama-3-8b-new/ \
    --dataset_name few_rel \
    --instruction_format_fn format_few_rel_pred \
    --max_new_tokens 15 \
    --response_key "### Response: "

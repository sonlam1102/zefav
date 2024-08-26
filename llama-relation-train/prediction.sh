python run_prediction.py \
    --model_path ./model/llama-3-8b-new/ \
    --dataset_path /home/sonlt/drive/data/programFC/feverous/new/dev.json \
    --max_new_tokens 80 \
    --response_key "### Response: " \
    --dataset_name "feverous" \
    --dataset_split "dev"

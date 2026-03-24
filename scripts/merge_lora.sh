#!/bin/bash
# Merge QLoRA adapters with base model for deployment
set -x

python -m openrlhf.cli.lora_combiner \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --lora_path ./models/rlvr \
    --output_path ./models/rlvr-merged \
    --param_dtype bf16

echo "Merged model saved to ./models/rlvr-merged"

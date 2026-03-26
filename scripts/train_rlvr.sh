#!/bin/bash
# GRPO + QLoRA training via OpenRLHF Ray
# No critic model — advantages from group-normalized rewards
set -x

MODEL="meta-llama/Llama-2-7b-chat-hf"
REWARD_FUNC="./rewards/reward_func.py"
SAVE_PATH="./models/rlvr"
PROMPT_DATA="./data/terminalbench_prompts.jsonl"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "."}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --pretrain $MODEL \
   --remote_rm_url $REWARD_FUNC \
   --save_path $SAVE_PATH \
   --prompt_data $PROMPT_DATA \
   --input_key "prompt" \
   --label_key "answer" \
   --apply_chat_template \
   --advantage_estimator group_norm \
   --n_samples_per_prompt 4 \
   --normalize_reward \
   --load_in_4bit \
   --lora_rank 64 \
   --lora_alpha 128 \
   --target_modules q_proj k_proj v_proj o_proj \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 128 \
   --max_samples 20000 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.001 \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps 100 \
   --use_wandb True \
   --wandb_project "afterquery" \
   --wandb_run_name "llama2-7b-chat-qlora-grpo"

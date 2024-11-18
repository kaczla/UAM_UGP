#!/usr/bin/env bash

export HF_HOME=.cache/hf
export TOKENIZERS_PARALLELISM='true'

rm -rf out/imdb-5k/gpt2

python run_glue.py \
  --cache_dir .cache_training \
  --model_name_or_path gpt2 \
  --use_lora 'True' \
  --lora_regex_pattern 'transformer[.]h[.][0-9]+[.](attn[.](c_proj|c_attn)|mlp[.](c_fc|c_proj))' \
  --lora_alpha 512 \
  --lora_r 256 \
  --train_file data/train-5k.json  \
  --validation_file data/valid-5k.json \
  --per_device_train_batch_size 24 \
  --per_device_eval_batch_size 24 \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 5 \
  --logging_strategy steps \
  --logging_steps 50 \
  --eval_steps 1000 \
  --evaluation_strategy steps \
  --metric_for_best_model 'accuracy' \
  --greater_is_better 'True' \
  --load_best_model_at_end 'True' \
  --report_to=none \
  --output_dir out/imdb-5k/gpt2_lora_4

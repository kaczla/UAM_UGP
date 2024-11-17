#!/usr/bin/env bash

export HF_HOME=.cache/hf
export TOKENIZERS_PARALLELISM='true'

rm -rf out/imdb-5k/t5_v1_0

python run_translation.py \
  --cache_dir .cache_training \
  --model_name_or_path "t5-small" \
  --train_file data/s2s-train-5k.json \
  --validation_file data/s2s-valid-5k.json \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --source_lang "text" \
  --target_lang "label" \
  --source_prefix "imdb classification" \
  --max_source_length 256 \
  --max_target_length 128 \
  --generation_max_length 128 \
  --do_train \
  --do_eval \
  --predict_with_generate \
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
  --output_dir out/imdb-5k/t5_v1_0

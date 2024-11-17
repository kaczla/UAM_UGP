#!/usr/bin/env bash

export HF_HOME=.cache/hf
export TOKENIZERS_PARALLELISM='true'

rm -rf out/imdb-5k/t5_v1_0

python run_translation.py \
  --cache_dir .cache_training \
  --model_name_or_path "t5-small" \
  --train_file data/s2s-train-5k.json \
  --validation_file data/s2s-valid-5k.json \
  --test_file data/s2s-test-5k.json \
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
  --do_predict \
  --predict_with_generate \
  --num_train_epochs 1 \
  --report_to=none \
  --output_dir out/imdb-5k/t5_v1_0

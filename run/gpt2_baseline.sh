#!/usr/bin/env bash

export HF_HOME=.cache/hf
export TOKENIZERS_PARALLELISM='true'

rm -rf out/imdb-5k/gpt2

python run_glue.py \
  --cache_dir .cache_training \
  --model_name_or_path gpt2 \
  --train_file data/train-5k.json  \
  --validation_file data/valid-5k.json \
  --test_file data/test-5k.json \
  --per_device_train_batch_size 24 \
  --per_device_eval_batch_size 24 \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --report_to=none \
  --output_dir out/imdb-5k/gpt2

#!/usr/bin/env bash

export HF_HOME=.cache/hf
export TOKENIZERS_PARALLELISM='true'

if [ "$#" -lt 3 ]; then
  echo >&2 'Missing model path and model save path! Example:'
  echo >&2 " bash $0 roberta-base out/imdb-5k/roberta_lora_1 out/imdb-5k/roberta_lora_1_merged"
  exit 1
fi

MODEL_NAME="$1"
MODEL_PATH="$2"
MODEL_SAVE="$3"

python merge_model.py \
  --base_model_name_or_path "${MODEL_NAME}" \
  --peft_model_name_or_path "${MODEL_PATH}" \
  --save_path "${MODEL_SAVE}" \
  --cache_dir .cache_training
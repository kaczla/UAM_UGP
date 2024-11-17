#!/usr/bin/env bash

export HF_HOME=.cache/hf
export TOKENIZERS_PARALLELISM='true'

if [ "$#" -lt 2 ]; then
  echo >&2 'Missing model path and model save path! Example:'
  echo >&2 " bash $0 out/imdb-5k/t5_v1_1 out/imdb-5k/t5_v1_1-evaluation"
  exit 1
fi

MODEL_PATH="$1"
MODEL_SAVE="$2"

python run_translation.py \
  --cache_dir .cache_training \
  --model_name_or_path "${MODEL_PATH}" \
  --train_file data/s2s-test-5k.json \
  --validation_file data/s2s-test-5k.json \
  --per_device_eval_batch_size 8 \
  --source_lang "text" \
  --target_lang "label" \
  --source_prefix "imdb classification" \
  --max_source_length 256 \
  --max_target_length 128 \
  --generation_max_length 128 \
  --do_eval \
  --predict_with_generate \
  --report_to=none \
  --output_dir "${MODEL_SAVE}"

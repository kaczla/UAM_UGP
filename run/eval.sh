#!/usr/bin/env bash

export HF_HOME=.cache/hf
export TOKENIZERS_PARALLELISM='true'

if [ "$#" -lt 2 ]; then
  echo >&2 'Missing model path and model save path! Example:'
  echo >&2 " bash $0 out/imdb-5k/roberta out/imdb-5k/roberta-evaluation"
  exit 1
fi

MODEL_PATH="$1"
MODEL_SAVE="$2"

python run_glue.py \
  --cache_dir .cache_training \
  --model_name_or_path "${MODEL_PATH}" \
  --train_file data/test-5k.json  \
  --validation_file data/test-5k.json \
  --per_device_eval_batch_size 24 \
  --do_eval \
  --max_seq_length 128 \
  --report_to=none \
  --output_dir "${MODEL_SAVE}"

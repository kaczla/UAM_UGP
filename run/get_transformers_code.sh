#!/usr/bin/env bash

TRANSFORMERS_VERSION='v4.46.2'

wget "https://raw.githubusercontent.com/huggingface/transformers/${TRANSFORMERS_VERSION}/examples/pytorch/text-classification/run_glue.py" -O 'run_glue.py'

wget "https://raw.githubusercontent.com/huggingface/transformers/${TRANSFORMERS_VERSION}/examples/pytorch/translation/run_translation.py" -O 'run_translation.py'

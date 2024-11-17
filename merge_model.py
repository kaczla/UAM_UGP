import logging
from dataclasses import dataclass, field
from typing import Optional

from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

LOGGER = logging.getLogger(__name__)


@dataclass
class MergeLoraArguments:
    base_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    peft_model_name_or_path: str = field(
        metadata={"help": "Path to PEFT model or model identifier from huggingface.co/models"}
    )
    save_path: str = field(
        metadata={"help": "Path where will be save merged model"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = HfArgumentParser((MergeLoraArguments,))
    merge_lora_arguments, = parser.parse_args_into_dataclasses()

    LOGGER.info(f"Loading base model: {merge_lora_arguments.base_model_name_or_path}")
    model = AutoModelForSequenceClassification.from_pretrained(merge_lora_arguments.base_model_name_or_path, cache_dir=merge_lora_arguments.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(merge_lora_arguments.base_model_name_or_path, cache_dir=merge_lora_arguments.cache_dir)

    LOGGER.info(f"Loading PEFT model: {merge_lora_arguments.peft_model_name_or_path}")
    model = PeftModel.from_pretrained(model, merge_lora_arguments.peft_model_name_or_path, cache_dir=merge_lora_arguments.cache_dir)

    LOGGER.info("Merging model")
    merged_model = model.merge_and_unload()
    LOGGER.info(f"Saving model in: {merge_lora_arguments.save_path}")
    merged_model.save_pretrained(merge_lora_arguments.save_path)
    tokenizer.save_pretrained(merge_lora_arguments.save_path)


if __name__ == '__main__':
    main()

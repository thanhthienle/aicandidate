"""Finetuning Gemma models for scoring ai-generated code."""

import logging
import sys
import re
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset

from unsloth import FastModel
from unsloth.chat_templates import (
    get_chat_template,
    train_on_responses_only
)

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from trl import SFTTrainer, SFTConfig

from sklearn.metrics import mean_squared_error

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.50.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


# To mark the code snippet of the answer.  For example: ```python\n#Beginning of code```
file_extension_to_markdown = {
    ".py": "python",
    ".swift": "swift",
    ".kt": "kotlin",
    ".cs": "csharp",
    ".dockerfile": "dockerfile",
    ".go": "go",
    ".cpp": "cpp",
    ".php": "php",
    ".ts": "typescript",
    ".js": "javascript",
    ".css": "css",
    ".sql": "sql",
    ".jav": "java",  # All java code in the dataset have extension '.jav', not '.java'
    ".dart": "dart"
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4 bit quantization to reduce memory."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8 bit quantization, which is a bit more accurate than 4 bit but uses 2x memory."},
    )
    full_finetuning: bool = field(
        default=False,
        metadata={"help": "Whether to use full finetuning."},
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "Rank of a LoRA module"},
    )
    lora_alpha: Optional[int] = field(
        default=8,
        metadata={"help": "The adapter matrix will be scaled by `lora_alpha/lora_r`, as mentioned in the LoRA paper by Hu et al."},
    )

    def __post_init__(self):
        # Quantization
        if self.load_in_4bit and self.load_in_8bit:
            logger.warning("Both 4-bit and 8-bit loading are detected as `True`. Prioritize 4-bit training, `load_in_8bit` will be override as `False`.")
            self.load_in_8bit = False

        # LoRA and Full Finetuning
        if self.lora_r and self.full_finetuning:
            logger.warning("LoRA rank detected yet the option for full finetuning is, currently, `True`. Prioritize LoRA finetuning, `full_finetuning` will be override as `False`.")
            self.full_finetuning = False


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    num_slices: Optional[int] = field(
        default=4, metadata={"help": "Divide the training set into `num_slices` slices, one of which will be used as validation split. Only effective when there isn't a dedicated validation split."}
    )

    def __post_init__(self):
        # Check train and test set
        if self.train_file is None and self.test_file is None:
            raise ValueError("Need a training/testing file.")
        else:
            if self.train_file is not None:
                train_extension = self.train_file.split(".")[-1]
                assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.test_file is not None:
                test_extension = self.test_file.split(".")[-1]
                if self.train_file is not None:
                    assert (
                        test_extension == train_extension
                    ), "`test_file` should have the same extension (csv or json) as `train_file`."

        # Check cross-validation
        if self.num_slices < 2:
            raise ValueError("Number of slices to divide the training set for validation purpose. Must should be at least 2.")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)


    # Loading a dataset from local files.
    # CSV/JSON training or testing files are needed.
    data_files = {}
    if data_args.train_file:
        data_files["train"] = data_args.train_file
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file:
        data_files["test"] = data_args.test_file
    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    # Loading a dataset from local json files
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        # token=model_args.token,
    )

    # Ensure the raw dataset contains the problem (question), code and label
    def check_data(dataset: datasets.Dataset):
        missing_keys = []
        missing_keys = [key for key in ("problem", "answer", "label", "extension") if key not in dataset.features]
        if missing_keys:
            raise KeyError(f"Missing required keys in dataset.features: {missing_keys}")

    # Converting datasets to the correct format for finetuning purposes
    def standardize_data_formats(examples: datasets.Dataset):
        prompt = (
            "Given the following coding problem and its solution, "
            "estimate the likelihood that the code was AI-generated. "
            "Provide a float score from 0 (entirely human-written) to 1 (entirely AI-generated).\n\n"
        )
        extension = file_extension_to_markdown[examples["extension"]]
        user_content = prompt + examples["problem"] + "\n\n\nANSWER:\n" + f"```{extension}\n" + examples["answer"] + "\n```"
        examples["conversations"] = [
            {"content": user_content, "role": "user"},
            {"content": str(examples["label"]), "role": "assistant"},
        ]
        return examples

    removed_keys = [key for key in raw_datasets["train"].column_names if key != "conversations"]
    for split in raw_datasets:
        check_data(raw_datasets[split])
        raw_datasets[split] = raw_datasets[split].map(standardize_data_formats, remove_columns=removed_keys)

    # Models and tokenizers with accelerate
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        max_seq_length = data_args.max_seq_length,
        load_in_4bit = model_args.load_in_4bit,    # 4 bit quantization to reduce memory
        load_in_8bit = model_args.load_in_8bit,    # Slightly more accurate, but costs 2x memory
        full_finetuning = model_args.full_finetuning,
    )

    # Apply the chat template for Gemma-3 onto the conversations, and save it to text
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )
    def apply_chat_template(examples):
        texts = tokenizer.apply_chat_template(examples["conversations"])
        return {"text": texts}
    for split in raw_datasets:
        raw_datasets[split] = raw_datasets[split].map(
            apply_chat_template, batched = True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )


    # Training
    if training_args.do_train:
        assert data_args.train_file, "Need training data to train."

        # LoRA
        if not model_args.full_finetuning:
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers     = False, # just text!
                finetune_language_layers   = True,
                finetune_attention_modules = True,
                finetune_mlp_modules       = True,

                r = model_args.lora_r,           # Larger = higher accuracy, but might overfit
                lora_alpha = model_args.lora_alpha,  # Recommended alpha == r at least
                lora_dropout = 0,
                bias = "none",
                random_state = training_args.seed,
            )

        if "validation" in data_files:
            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["eval"]
        elif training_args.do_eval:
            dataset = raw_datasets["train"].shuffle(seed=training_args.seed)
            dataset_size = len(dataset)
            fold_size = dataset_size // data_args.num_slices

            # Create validation and train splits
            eval_dataset = dataset.select(range(fold_size))
            train_dataset = dataset.select(range(fold_size, dataset_size))

        # Data Collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = training_args.per_device_train_batch_size,
                gradient_accumulation_steps = training_args.gradient_accumulation_steps, # Use GA to mimic batch size!
                warmup_steps = training_args.warmup_steps,
                max_steps = training_args.max_steps,
                learning_rate = training_args.learning_rate, # Reduce to 2e-5 for long training runs
                logging_steps = training_args.logging_steps,
                optim = training_args.optim,
                weight_decay = training_args.weight_decay,
                seed = training_args.seed,
                save_strategy=training_args.save_strategy,
                output_dir=training_args.output_dir,
                report_to = "none", # Use this for WandB etc
                max_seq_length = data_args.max_seq_length,
                do_eval=training_args.do_eval
            ),
            data_collator=data_collator,
        )

        # only train on the assistant outputs and ignore the loss on the user's inputs.
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<start_of_turn>user\n",
            response_part = "<start_of_turn>model\n",
        )

        # Train
        trainer_stats = trainer.train()
        trainer.save_model()
        metrics = trainer_stats.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if training_args.do_eval:
            eval_dataset = eval_dataset.rename_column("text", "label")
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

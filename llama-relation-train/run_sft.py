from dataclasses import dataclass, field
import dataclasses
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
import os
import json

from trl import SFTTrainer, is_xpu_available
import importlib

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(
        default="/scr/huggingface_models/Llama-2-7b-hf",
        metadata={"help": "the model name"},
    )
    dataset_name: Optional[str] = field(
        default="yangwang825/marc-ja", metadata={"help": "the dataset name"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_split: Optional[str] = field(
        default=None, 
        metadata={"help": "the dataset spilt (train / dev)"}
    )
    train_file: Optional[str] = field(
        default="data/train.jsonl",
        metadata={"help": "the dataset file"},
    )
    test_file: Optional[str] = field(
        default="data/val.jsonl",
        metadata={"help": "the dataset file"},
    )
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )
    use_debug_example: Optional[int] = field(
        default=None,
        metadata={"help": "Only use some examples for debugging"},
    )
    seq_length: Optional[int] = field(
        default=1024, metadata={"help": "Input sequence length"}
    )
    bits: Optional[int] = field(
        default=None, metadata={"help": "How many bits to use."}
    )
    use_peft: Optional[bool] = field(
        default=True, metadata={"help": "Wether to use PEFT or not to train adapters"}
    )
    trust_remote_code: Optional[bool] = field(
        default=False, metadata={"help": "Enable `trust_remote_code`"}
    )
    peft_lora_r: Optional[int] = field(
        default=16, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=32, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    use_auth_token: Optional[bool] = field(
        default=True, metadata={"help": "Use HF auth token to access the model"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to use flash attention 2")},
    )
    instruction_format_fn: str = field(
        default=False,
        metadata={"help": ("The instruction template to map training example to text")},
    )


parser = HfArgumentParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()


def import_inst_format_fn(fn_name):
    try:
        inst_util_module = "instruction_util"
        module = importlib.import_module(inst_util_module)

        # Try to get the specified function from the module
        try:
            fn = getattr(module, fn_name)
            # Now you can use the function as needed
            print(
                f"Function '{fn_name}' from module '{inst_util_module}' successfully imported."
            )
            return fn
        except AttributeError:
            print(f"Function '{fn_name}' not found in module '{inst_util_module}'.")
    except ImportError:
        print(f"Could not import module '{inst_util_module}'.")


inst_format_fn = import_inst_format_fn(script_args.instruction_format_fn)

os.makedirs(training_args.output_dir, exist_ok=True)

with open(os.path.join(training_args.output_dir, "experiment_args.json"), "w") as f:
    exp_args = dataclasses.asdict(training_args)
    exp_args.update(dataclasses.asdict(script_args))
    json.dump(exp_args, f, indent=4, ensure_ascii=False)


# Step 1: Load the model
if script_args.bits:
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=script_args.bits == 4,
        load_in_8bit=script_args.bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=script_args.double_quant,
        bnb_4bit_quant_type=script_args.quant_type,
    )
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

torch_dtype = (
    script_args.torch_dtype
    if script_args.torch_dtype in ["auto", None]
    else getattr(torch, script_args.torch_dtype)
)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    load_in_4bit=script_args.bits == 4,
    load_in_8bit=script_args.bits == 8,
    quantization_config=quantization_config,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    token=script_args.use_auth_token,
    low_cpu_mem_usage=script_args.low_cpu_mem_usage,
    device_map="auto",
    use_flash_attention_2=script_args.use_flash_attention_2,
)

# Step 2: Load the dataset
if script_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        script_args.dataset_name,
        script_args.dataset_config_name,
        cache_dir=script_args.cache_dir,
    )
else:
    data_files = {}
    if script_args.train_file is not None:
        data_files["train"] = script_args.train_file
        extension = script_args.train_file.split(".")[-1]
    if script_args.validation_file is not None:
        data_files["validation"] = script_args.validation_file
        extension = script_args.validation_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=script_args.cache_dir,
        token=script_args.token,
    )


def make_text(example):
    example[script_args.dataset_text_field] = inst_format_fn(example)
    return example


dataset = raw_datasets.map(make_text)

print(f"Some training examples:")
for d in dataset["train_wiki"].select(range(5)):
    print(json.dumps(d, ensure_ascii=False))
    print("-" * 10)

if script_args.use_debug_example is not None:
    dataset["train_wiki"] = dataset["train_wiki"].select(range(script_args.use_debug_example))
    dataset["val_wiki"] = dataset["val_wiki"].select(
        range(script_args.use_debug_example)
    )

# Step 4: Define the LoraConfig
if script_args.use_peft:
    print("Training using PEFT")
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    print("Training full model (No PEFT)")
    peft_config = None

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset["train_wiki"],
    eval_dataset=dataset["val_wiki"],
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
)

trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

# Step 6: Save the model
trainer.save_model(training_args.output_dir)

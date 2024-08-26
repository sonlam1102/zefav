import os
import json
import fire
from tqdm import tqdm
import torch
from datasets import load_dataset
from peft import PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import importlib

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")


def load_peft_model(peft_model_name, **kwargs):
    config = PeftConfig.from_pretrained(peft_model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        model_max_length=1024,
        padding_side="left",
        truncation_side="left",
        token=os.getenv("HF_ACCESS_TOKEN"),
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        token=os.getenv("HF_ACCESS_TOKEN"),
    )
    return tokenizer, model


@torch.inference_mode()
def do_inference(model, tokenizer, prompt, max_new_tokens=256, add_special_tokens=True):
    token_ids = tokenizer.encode(
        prompt, add_special_tokens=add_special_tokens, return_tensors="pt", truncation=True
    )
    output_ids = model.generate(
        input_ids=token_ids.to(model.device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.batch_decode(output_ids)[0]


def import_inst_format_fn(fn_name):
    try:
        inst_util_module = "instruction_util_predict"
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


def main(
    model_path,
    dataset_name,
    instruction_format_fn,
    override_result=False,
    max_new_tokens=256,
    tokenizing_add_special_tokens=True,
    text_field="text",
    response_key="### Response: ",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res_output = os.path.join(model_path, "test.json")
    if os.path.exists(res_output) and not override_result:
        print("Eval result already exists, ", res_output)
        return

    tokenizer, model = load_peft_model(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)

    inst_format_fn = import_inst_format_fn(instruction_format_fn)

    def make_text(example):
        example[text_field] = inst_format_fn(example)
        return example

    dataset = load_dataset(dataset_name, "default")["val_wiki"]
    dataset = dataset.map(make_text, load_from_cache_file=False)

    result = []
    for example in tqdm(dataset):
        prompt = example[text_field]
        pred = do_inference(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            add_special_tokens=tokenizing_add_special_tokens,
        )
        pred = pred.split(response_key)[1]
        result.append({**example, "output_prediction": pred})

    with open(os.path.join(model_path, "output_prediction.json"), "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)

import os
import json
import fire
from tqdm import tqdm
import torch
# from datasets import load_dataset
from peft import PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import importlib
from nltk.tokenize import word_tokenize, sent_tokenize

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")


def format_hover_pred(example):
    return f"""### Instruction: Given a sentence, please identify the head and tail entities in the sentence, 
    and classify the relation type into one of appropriate categories; The collection of categories is: [ 
    screenwriter, has part, said to be the same as, composer, participating team, headquarters location, 
    heritage designation, after a work by, participant, part of, performer, work location, operating system, 
    instance of, original language of film or TV show, follows, country of citizenship, residence, architect, 
    position held, genre, original network, main subject, sport, mountain range, publisher, manufacturer, 
    located on terrain feature, instrument, country of origin, position played on team / speciality, 
    developer, military branch, movement, distributor, owned by, platform, 
    located in or next to body of water, nominated for, location, place served by transport hub, 
    league, religion, military rank, successful candidate, operator, country, sibling, mouth of the watercourse, 
    constellation, child, notable work, field of work, subsidiary, winner, director, crosses, 
    member of political party, licensed to broadcast to, tributary, location of formation, 
    spouse, sports season of league or competition, language of work or name, occupation, 
    head of government, occupant, mother, competition class, located in the administrative territorial entity, 
    contains administrative territorial entity, participant of, voice type, followed by, member of, father, 
    record label, taxon rank, characters, applies to jurisdiction]; 
    Sentence: {example}\n ### Response: """


def load_peft_model(peft_model_name, **kwargs):
    config = PeftConfig.from_pretrained(peft_model_name)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        model_max_length=1024,
        padding_side="left",
        truncation_side="left",
        token="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",
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        token="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",
        # quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        device_map="cuda:1"
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
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.batch_decode(output_ids)[0]


def import_inst_format_fn(fn_name):
    try:
        inst_util_module = "instruction_util_prediction"
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
    dataset_path,
    override_result=False,
    max_new_tokens=256,
    tokenizing_add_special_tokens=True,
    response_key="### Response: ",
    dataset_name="feverous",
    dataset_split="train"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res_output = os.path.join(model_path, "test.json")
    if os.path.exists(res_output) and not override_result:
        print("Eval result already exists, ", res_output)
        return

    tokenizer, model = load_peft_model(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path)
    # model.to(device)

    # inst_format_fn = format_hover_pred(instruction_format_fn)

    def make_text(example):
        # example[text_field] = inst_format_fn(example)
        return format_hover_pred(example)

    # dataset = load_dataset(dataset_name, "default")["val_wiki"]
    # dataset = dataset.map(make_text, load_from_cache_file=False)
    with open(dataset_path, 'r') as json_data:
        dataset = json.load(json_data)

    result = []
    for example in tqdm(dataset):
        prompt_claim = make_text(example['claim'])
        pred_claim = do_inference(
            model,
            tokenizer,
            prompt_claim,
            max_new_tokens=max_new_tokens,
            add_special_tokens=tokenizing_add_special_tokens,
        )
        pred_claim = pred_claim.split(response_key)[1]

        # list_sentence_evidence = example['evidence'].split('\n')
        if dataset_name == 'feverous':
            # print(dataset_name)
            # list_sentence_evidence = sent_tokenize(example['evidence_new'])
            list_sentence_evidence = example['evidence_new'].split('\n')
        else:
            list_sentence_evidence = sent_tokenize(example['evidence'])
        
        list_re_e = []
        for e in list_sentence_evidence:
            try:
                prompt_e = make_text(e.split("\t")[1] if dataset_name == 'feverous' else e)
            except Exception as err:
                print(err)
                print("------")
                print(e)
                prompt_e = make_text(e)

            pred_e = do_inference(
                model,
                tokenizer,
                prompt_e,
                max_new_tokens=max_new_tokens,
                add_special_tokens=tokenizing_add_special_tokens,
            )
            pred_e = pred_e.split(response_key)[1]
            list_re_e.append(pred_e)
        
        result.append({**example, "claim_rels": pred_claim, "evidence_rels": list_re_e})


    with open(os.path.join(model_path, "output_prediction_{}_{}.json".format(dataset_name, dataset_split)), "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)

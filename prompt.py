import json
import re

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import MvpTokenizer, MvpForConditionalGeneration
from tqdm import tqdm


def format_hover_with_infore(query, evidence_rel, claim_rel, infore, full_context):
    infore_str = ""
    for ir in infore:
        infore_str = infore_str + ir + "\n\t\t"

    s_sample2 = ""
    k2 = 1

    for ev in claim_rel:
        s_sample2 = s_sample2 + "{}. The question mentioned the relation between \"{}\" and \"{}\" as \"{}\".\n\t\t".format(k2, ev[0].strip(),
                                                                                                                        ev[2].strip(),
                                                                                                                        ev[1].strip())
        k2 = k2 + 1

    for l in evidence_rel:
        s_sample2 = s_sample2 + "{}. \"{}\" has relation with \"{}\" as \"{}\".\n\t\t".format(k2, l[0].strip(),
                                                                                              l[2].strip(),
                                                                                              l[1].strip())
        k2 = k2 + 1

    return f"""
            Documents: 
                {infore_str} \n
            Context: 
                {full_context} \n
            Question: {query}?\n
            Please answer the question based on Documents, Context and the following relations. The answer must belong to one of two values: True or False.
                {s_sample2}
            Let's think step-by-step.
            ###The answer is:  
    """


def format_hover_with_infore_no_relation(query, infore, full_context):
    infore_str = ""
    for ir in infore:
        infore_str = infore_str + ir + "\n\t\t"

    return f"""
            Documents: 
                {infore_str} \n
            Context: 
                {full_context} \n
            Question: {query}?\n
            Please answer the question based on Documents and Context. The answer must belong to one of two values: True or False.
            Let's think step-by-step.
            ###The answer is:  
    """


def format_hover_with_infore_no_context(query, evidence_rel, claim_rel, infore):
    infore_str = ""
    for ir in infore:
        infore_str = infore_str + ir + "\n\t\t"

    s_sample2 = ""
    k2 = 1

    for ev in claim_rel:
        s_sample2 = s_sample2 + "{}. The question mentioned the relation between \"{}\" and \"{}\" as \"{}\".\n\t\t".format(k2, ev[0].strip(),
                                                                                                                        ev[2].strip(),
                                                                                                                        ev[1].strip())
        k2 = k2 + 1

    for l in evidence_rel:
        s_sample2 = s_sample2 + "{}. \"{}\" has relation with \"{}\" as \"{}\".\n\t\t".format(k2, l[0].strip(),
                                                                                              l[2].strip(),
                                                                                              l[1].strip())
        k2 = k2 + 1

    return f"""
            Documents: 
                {infore_str} \n
            Question: {query}?\n
            Please answer the question based on Documents and the following relations. The answer must belong to one of two values: True or False.
                {s_sample2}
            Let's think step-by-step.
            ###The answer is:  
    """


def format_hover_with_infore_no_infore(query, evidence_rel, claim_rel, full_context):
    s_sample2 = ""
    k2 = 1

    for ev in claim_rel:
        s_sample2 = s_sample2 + "{}. The question mentioned the relation between \"{}\" and \"{}\" as \"{}\".\n\t\t".format(k2, ev[0].strip(),
                                                                                                                        ev[2].strip(),
                                                                                                                        ev[1].strip())
        k2 = k2 + 1

    for l in evidence_rel:
        s_sample2 = s_sample2 + "{}. \"{}\" has relation with \"{}\" as \"{}\".\n\t\t".format(k2, l[0].strip(),
                                                                                              l[2].strip(),
                                                                                              l[1].strip())
        k2 = k2 + 1

    return f"""
            Context: 
                {full_context} \n
            Question: {query}?\n
            Please answer the question based on Context and the following relations. The answer must belong to one of two values: True or False.
                {s_sample2}
            Let's think step-by-step.
            ###The answer is:  
    """


def format_hover_with_infore_no_context_no_relation(query, infore):
    infore_str = ""
    for ir in infore:
        infore_str = infore_str + ir + "\n\t\t"

    return f"""
            Documents: 
                {infore_str} \n
            Question: {query}?\n
            Please answer the question based on Documents. The answer must belong to one of two values: True or False.
            Let's think step-by-step.
            ###The answer is:  
    """


def format_hover_with_infore_no_context_no_infore(query, evidence_rel, claim_rel):
    s_sample2 = ""
    k2 = 1

    for ev in claim_rel:
        s_sample2 = s_sample2 + "{}. The question mentioned the relation between \"{}\" and \"{}\" as \"{}\".\n\t\t".format(k2, ev[0].strip(),
                                                                                                                        ev[2].strip(),
                                                                                                                        ev[1].strip())
        k2 = k2 + 1

    for l in evidence_rel:
        s_sample2 = s_sample2 + "{}. \"{}\" has relation with \"{}\" as \"{}\".\n\t\t".format(k2, l[0].strip(),
                                                                                              l[2].strip(),
                                                                                              l[1].strip())
        k2 = k2 + 1

    return f"""
            Question: {query}?\n
            Please answer the question based on the following relations. The answer must belong to one of two values: True or False.
                {s_sample2}
            Let's think step-by-step.
            ###The answer is:  
    """


def format_feverous_with_infore(query, evidence_rel, claim_rel, infore, full_context):
    s_sample2 = ""
    k2 = 1

    for ev in claim_rel:
        s_sample2 = s_sample2 + "{}. The question mentioned the relation between \"{}\" and \"{}\" as \"{}\".\n\t\t".format(k2, ev[0].strip(),
                                                                                                                        ev[2].strip(),
                                                                                                                        ev[1].strip())
        k2 = k2 + 1

    for l in evidence_rel:
        s_sample2 = s_sample2 + "{}. \"{}\" has relation with \"{}\" as \"{}\".\n\t\t".format(k2, l[0].strip(),
                                                                                              l[2].strip(),
                                                                                              l[1].strip())
        k2 = k2 + 1

    return f"""
            Documents: 
                {infore} \n
            Context in tabular: 
                {full_context} \n
            Question: {query}?\n
            Please answer the question based on Documents, Tabular Context and the following relations. The answer must belong to one of two values: True or False.
                {s_sample2}
            Let's think step-by-step.
            ###The answer is:  
    """


def format_feverous_with_infore_no_contex(query, evidence_rel, claim_rel, infore):
    s_sample2 = ""
    k2 = 1

    for ev in claim_rel:
        s_sample2 = s_sample2 + "{}. The question mentioned the relation between \"{}\" and \"{}\" as \"{}\".\n\t\t".format(k2, ev[0].strip(),
                                                                                                                        ev[2].strip(),
                                                                                                                        ev[1].strip())
        k2 = k2 + 1

    for l in evidence_rel:
        s_sample2 = s_sample2 + "{}. \"{}\" has relation with \"{}\" as \"{}\".\n\t\t".format(k2, l[0].strip(),
                                                                                              l[2].strip(),
                                                                                              l[1].strip())
        k2 = k2 + 1

    return f"""
            Documents: 
                {infore} \n
            Question: {query}?\n
            Please answer the question based on Documents and the following relations. The answer must belong to one of two values: True or False.
                {s_sample2}
            Let's think step-by-step.
            ###The answer is:  
    """


def format_feverous_with_infore_no_contex_no_relation(query, infore):
    return f"""
            Documents: 
                {infore} \n
            Question: {query}?\n
            Please answer the question based on Documents. The answer must belong to one of two values: True or False.
            
            Let's think step-by-step.
            ###The answer is:  
    """


def format_feverous_with_infore_no_contex_no_infore(query, evidence_rel, claim_rel):
    s_sample2 = ""
    k2 = 1

    for ev in claim_rel:
        s_sample2 = s_sample2 + "{}. The question mentioned the relation between \"{}\" and \"{}\" as \"{}\".\n\t\t".format(k2, ev[0].strip(),
                                                                                                                        ev[2].strip(),
                                                                                                                        ev[1].strip())
        k2 = k2 + 1

    for l in evidence_rel:
        s_sample2 = s_sample2 + "{}. \"{}\" has relation with \"{}\" as \"{}\".\n\t\t".format(k2, l[0].strip(),
                                                                                              l[2].strip(),
                                                                                              l[1].strip())
        k2 = k2 + 1

    return f"""
            Question: {query}?\n
            Please answer the question based on the following relations. The answer must belong to one of two values: True or False.
                {s_sample2}
            Let's think step-by-step.
            ###The answer is:  
    """


def format_feverous_with_infore_no_relation(query, infore, full_context):
    return f"""
            Documents: 
                {infore} \n
            Context in tabular: 
                {full_context} \n
            Question: {query}?\n
            Please answer the question based on Documents and Tabular Context. The answer must belong to one of two values: True or False.
            Let's think step-by-step.
            ###The answer is:  
    """


def format_feverous_with_infore_no_infore(query, evidence_rel, claim_rel, full_context):
    s_sample2 = ""
    k2 = 1

    for ev in claim_rel:
        s_sample2 = s_sample2 + "{}. The question mentioned the relation between \"{}\" and \"{}\" as \"{}\".\n\t\t".format(k2, ev[0].strip(),
                                                                                                                        ev[2].strip(),
                                                                                                                        ev[1].strip())
        k2 = k2 + 1

    for l in evidence_rel:
        s_sample2 = s_sample2 + "{}. \"{}\" has relation with \"{}\" as \"{}\".\n\t\t".format(k2, l[0].strip(),
                                                                                              l[2].strip(),
                                                                                              l[1].strip())
        k2 = k2 + 1

    return f"""
            Context in tabular: 
                {full_context} \n
            Question: {query}?\n
            Please answer the question based on Tabular Context and the following relations. The answer must belong to one of two values: True or False.
                {s_sample2}
            Let's think step-by-step.
            ###The answer is:  
    """


def load_peft_model(peft_model_name, device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(
        peft_model_name,
        model_max_length=2048,
        padding_side="left",
        truncation_side="left",
        token="<Please fill your Huggingface access Token here>"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        # load_in_4bit=False,
        # load_in_8bit=True,
        quantization_config=quantization_config,
        token="<Please fill your Huggingface access Token here>",
        use_flash_attention_2=False,
        device_map=device
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
        top_p=0.6,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.batch_decode(output_ids)[0]


def perform_prompt(split="dev", name="hover", device="auto"):
    tokenizer, model = load_peft_model("meta-llama/Meta-Llama-3-70B-Instruct", device)
    # tokenizer, model = load_peft_model("mistralai/Mixtral-8x7B-v0.1")

    # with open('dump/output_new_verbalized_{}.json'.format(split), 'r') as f:
    # with open('dump/output_new_{}.json'.format(split), 'r') as f:
    with open('dump/output_new_infore_{}_{}.json'.format(split, name), 'r') as f:
        dataset = json.load(f)
    f.close()

    demo = dataset[0]
    if name == "hover":
        prompt_claim = format_hover_with_infore(demo['claim'], demo['retrieved_evidence'], demo['claim_relation'], demo['evidence_reorganized'], demo['evidence'])
        # prompt_claim = format_hover_with_infore_no_context(demo['claim'], demo['retrieved_evidence'], demo['claim_relation'], demo['evidence_reorganized'])
        # prompt_claim = format_hover_with_infore_no_relation(demo['claim'], demo['evidence_reorganized'], demo['evidence'])
        # prompt_claim = format_hover_with_infore_no_infore(demo['claim'], demo['retrieved_evidence'], demo['claim_relation'], demo['evidence'])
        # prompt_claim = format_hover_with_infore_no_context_no_relation(demo['claim'], demo['evidence_reorganized'])
        # prompt_claim = format_hover_with_infore_no_context_no_infore(demo['claim'], demo['retrieved_evidence'], demo['claim_relation'])
    else:
        prompt_claim = format_feverous_with_infore(demo['claim'], demo['retrieved_evidence'], demo['claim_relation'], demo['evidence_reorganized'], demo['evidence_new'])
        # prompt_claim = format_feverous_with_infore_no_infore(demo['claim'], demo['retrieved_evidence'], demo['claim_relation'], demo['evidence_new'])
        # prompt_claim = format_feverous_with_infore_no_relation(demo['claim'], demo['evidence_reorganized'], demo['evidence_new'])
        # prompt_claim = format_feverous_with_infore_no_contex(demo['claim'], demo['retrieved_evidence'], demo['claim_relation'], demo['evidence_reorganized'])
        # prompt_claim = format_feverous_with_infore_no_contex_no_relation(demo['claim'], demo['evidence_reorganized'])
        # prompt_claim = format_feverous_with_infore_no_contex_no_infore(demo['claim'], demo['retrieved_evidence'], demo['claim_relation'])
    print(prompt_claim)

    result = []
    for example in tqdm(dataset):
        if name == "hover":
            prompt_claim = format_hover_with_infore(example['claim'], example['retrieved_evidence'], example['claim_relation'], example['evidence_reorganized'], example['evidence'])
            # prompt_claim = format_hover_with_infore_no_infore(example['claim'], example['retrieved_evidence'], example['claim_relation'], example['evidence'])
            # prompt_claim = format_hover_with_infore_no_relation(example['claim'], example['evidence_reorganized'], example['evidence'])
            # prompt_claim = format_hover_with_infore_no_context(example['claim'], example['retrieved_evidence'], example['claim_relation'], example['evidence_reorganized'])
            # prompt_claim = format_hover_with_infore_no_context_no_relation(example['claim'], example['evidence_reorganized'])
            # prompt_claim = format_hover_with_infore_no_context_no_infore(example['claim'], example['retrieved_evidence'], example['claim_relation'])
        else:
            prompt_claim = format_feverous_with_infore(example['claim'], example['retrieved_evidence'], example['claim_relation'], example['evidence_reorganized'], example['evidence_new'])
            # prompt_claim = format_feverous_with_infore_no_infore(example['claim'], example['retrieved_evidence'], example['claim_relation'], example['evidence_new'])
            # prompt_claim = format_feverous_with_infore_no_relation(example['claim'], example['evidence_reorganized'], example['evidence_new'])
            # prompt_claim = format_feverous_with_infore_no_contex(example['claim'], example['retrieved_evidence'], example['claim_relation'], example['evidence_reorganized'])
            # prompt_claim = format_feverous_with_infore_no_contex_no_relation(example['claim'], example['evidence_reorganized'])
            # prompt_claim = format_feverous_with_infore_no_contex_no_infore(example['claim'], example['retrieved_evidence'], example['claim_relation'])

        pred_claim = do_inference(
            model,
            tokenizer,
            prompt_claim,
            max_new_tokens=2,
            add_special_tokens=False,
        )
        pred_claim = pred_claim
        result.append({**example, "predicted_label": pred_claim})

    with open("dump/output_prediction_prompt_{}_{}.json".format(split, name), "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


def retrieve_rel_text(data):
    m = re.search(r'(\(.*?[A-z|a-z|0-9].*\))', data)
    return m


def change_to_tuple(string):
    try:
        string = string.strip('(')
        string = string.strip(')')
        list_ele = string.split(',')
        return (list_ele[0], list_ele[1], list_ele[2])
    except Exception as e:
        return None


def retrieve_rels(data):
    data = data.replace(" - ", "")
    data = data.replace("[", "(")
    data = data.replace("]", ")")
    data = data.replace("( ", "(")
    data = data.replace(" )", ")")
    try:
        fetch = retrieve_rel_text(data)
        list_rel = fetch.group(0).split("  ")
        list_rel = list(set(list_rel))
        # list_rel = [change_to_tuple(a) for a in list_rel]
        for a in list_rel:
            if change_to_tuple(a) is not None:
                list_rel.append(change_to_tuple(a))
    except Exception as e:
        return []
    return list_rel


def find_math(e, list):
    for l in list:
        if re.search(r'{}'.format(e), l):
            return True
    return False


def inductive_reasoning(claim, evidence):
    claim_rel = retrieve_rels(claim)
    evidence_rel = []
    for le in evidence:
        result = retrieve_rels(le)
        if result is not None:
            evidence_rel.extend(result)
    hypos = []
    retrieve_rel = []
    claim_rel = [cl for cl in claim_rel if not isinstance(cl, str)]
    if len(claim_rel) > 0:
        try:
            for c in claim_rel:
                hypos.append(c[0])
                hypos.append(c[2])
            hypos = list(set(hypos))
        except Exception as e:
            print(e)
            pass

    is_found = True
    len_hypos = len(hypos)
    while is_found:
        for e in evidence_rel:
            try:
                if find_math(e[0].strip(), hypos):
                    hypos.append(e[2])
                    # hypos.append(e[0])
                    retrieve_rel.append(e)

                    hypos = list(set(hypos))
                    retrieve_rel = list(set(retrieve_rel))
            except Exception as er:
                pass
        if len_hypos == len(hypos):
            is_found = False
        else:
            len_hypos = len(hypos)

    return retrieve_rel, claim_rel


def operate_induct_hover(data, split="dev", dname="hover"):
    for d in data:
        list_rel, claim_rel = inductive_reasoning(d['claim_rels'], d['evidence_rels'])
        d['retrieved_evidence'] = list_rel
        d['claim_relation'] = claim_rel
    with open("dataset/output_new_{}_{}.json".format(split, dname), 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def triple_generation(data, split="dev"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp")
    model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text")
    model = model.to(device)
    for d in tqdm(data):
        verbal_evd = []
        for l in d['retrieved_evidence']:
            prompt = "Describe the following data: {} | {} | {}".format(l[0], l[1], l[2])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs, max_length=100)
            verbal_evd.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
        d['retrieved_verbalized_evidence'] = verbal_evd

    with open("dump/output_new_verbalized_{}.json".format(split), 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def pre_process_feverous_evidence(string):
    tab_vals = re.search(r"\[\[\b([a-zA-Z0-9&_| |\-\.\,\(\)\'| -*])*\b\]\]", string)
    while tab_vals is not None:
        extracted = tab_vals.group(0)
        s1 = string.replace(extracted, "<sub>")
        string = s1.replace("<sub>", extracted.split("|")[-1].replace("]]", ""))
        tab_vals = re.search(r"\[\[\b([a-zA-Z0-9&_| |\-\.\,\(\)\'| -*])*\b\]\]", string)
    return string


if __name__ == '__main__':
    print("-----Choices for Prompting on ZeFAV------\n")
    screen_message = """
    1. Performing ZeFAV \n
    2. Inducting relation \n
    3. Triplet generation (not available) \n
    4. Pre-process feverous \n 
    5. Integration InfoRE to original \n 
    """
    print(screen_message)

    choice = input("Choose options: ")
    choice = int(choice)
    if choice == 1:
        # PERFORM PROMPTING
        print("Running ZeFAV ....")
        perform_prompt(split="dev", name="hover", device="auto")
    elif choice == 2:
        print("Running induction...")
        # INDUCTION RELATION
        with open("dataset/output_prediction_feverous_dev.json", 'r') as f:
            dev_pred = json.load(f)
        f.close()
        # print(len(dev_pred))
        operate_induct_hover(dev_pred, split="dev", dname="feverous")

        with open("dataset/output_prediction_hover_dev.json", 'r') as f:
            train_pred = json.load(f)
        f.close()
        operate_induct_hover(train_pred, split="dev")
    elif choice == 3:
        # print("Generating triplet ....")
        # # triplet generation
        # with open("dump/output_new_dev.json", 'r') as f:
        #     dev_pred = json.load(f)
        # f.close()
        # triple_generation(dev_pred, split="dev")
        pass
    elif choice == 4:
        print("Pre-processing FeVEROUS ....")
        # Pre-process feverous
        with open("dataset/feverous/dev.json", 'r') as f:
            dev_pred = json.load(f)
        f.close()
        for devf in tqdm(dev_pred):
            devf['evidence_new'] = pre_process_feverous_evidence(devf['evidence'])
        with open("dataset/feverous/dev_new.json", 'w') as f:
            json.dump(dev_pred, f, ensure_ascii=False, indent=4)
    elif choice == 5:
        print("Integrating InfoRE ....")

        # integration InfoRE to original
        with open('dataset/output_new_dev_feverous.json', 'r') as f:
            dataset = json.load(f)
        f.close()

        with open('dataset/reorganized_claims_dev_feverous.json', 'r') as f:
            dataset_infore = json.load(f)
        f.close()

        for i in range(0, len(dataset)):
            assert len(dataset_infore) == len(dataset)
            assert dataset[i]['id'] == dataset_infore[i]['id']
            dataset[i]['evidence_reorganized'] = dataset_infore[i]['evidence']

        with open("dataset/output_new_infore_dev_feverous.json", 'w') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
    else:
        print("Not match!")

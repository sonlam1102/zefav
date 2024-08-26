from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
from tqdm import tqdm

template = """Given a claim and corresponding evidence, please summarize the evidence according to the claim. The summarized evidence should contain information that could helps support or refute the claim. Here is an example:
CLAIM: Skagen Painter Peder Severin Krøyer favored naturalism along with Theodor Esbern Philipsen and the artist Ossian Elgström studied with in the early 1900s.
EVIDENCE:
Ossian Elgström (1883 – 1950) was a Swedish illustrator and writer. He was a brother of writer and visual artist Anna Lenah Elgström. Elgström studied at the Royal Swedish Academy of Arts from 1906 to 1907, and then with Kristian Zahrtmann in 1907 and with Christian Krohg in 1908. He contributed to the magazines "Strix", "Söndags-Nisse" and "Puck". He collected folkloristic material from Siberia, Greenland and Lappland, which he used in his books. Among his books are "Lapska myther" from 1914, "Lappalaiset" from 1919, and "Karesuando-lapparna" from 1922.
Peder Severin Krøyer [\'peːdə \'seveʁin \'kʁojə] (23 July 1851 – 21 November 1909), professionally known as P. S. Krøyer, was a Danish painter. He is one of the best known and beloved, and the most colorful of the Skagen Painters, a community of Danish and Nordic artists who lived, gathered, or worked in Skagen, Denmark, especially during the final decades of the 19th century. Krøyer was the unofficial leader of the group.
Peder Henrik Kristian Zahrtmann, known as Kristian Zahrtmann, (31 March 1843 – 22 June 1917) was a Danish painter. He was a part of the Danish artistic generation in the late 19th century, along with Peder Severin Krøyer and Theodor Esbern Philipsen, who broke away from both the strictures of traditional Academicism and the heritage of the Golden Age of Danish Painting, in favor of naturalism and realism.
MIND MAP:
Ossian Elgström:
    Education:
        School: Royal Swedish Academy of Arts
        Year: from 1906 to 1907
        Companion 1:
            Name: Kristian Zahrtmann
            Year: 1908
        Companion 2:
            Name: Christian Krohg
            Year: 1908
Peder Severin Krøyer:
    Profession: Painter
    Nationality: Danish
    Group affiliation: Skagen Painters
Kristian Zahrtmann:
    Artistic generation:
        Description: late 19th century
        Fellow artists:
            Name 1: Peder Severin Krøyer
            Name 2: Theodor Esbern Philipsen
        Artistic style:
            Break away from:
                Academicism
                Golden Age of Danish Painting
            Embraced:
                Naturalism
                Realism
###
CLAIM: {}
EVIDENCE: 
{}
MIND MAP:"""

model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.float16, use_flash_attention_2=True
)


def build_prompt(claim, evidence):
    return template.format(claim, evidence)


def generate_text(prompt, max_new_tokens=128):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0,
            do_sample=False,
        )
        output_text = tokenizer.decode(
            outputs[0][len(inputs[0]) :], skip_special_tokens=True
        )
    return output_text


def reorganize_evidence(claim, evidence):
    prompt = build_prompt(claim, evidence)
    output = generate_text(prompt, max_new_tokens=128)
    reorganized_evidence = output.split("\n")[0].split(".")[0].strip()
    return reorganized_evidence


def load_dataset(claim_file):
    with open(claim_file) as f:
        claims = json.load(f)
    return claims


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = load_args()
    claims = load_dataset(args.claim_file)
    outputs = []
    with tqdm(total=len(claims)) as pbar:
        for i, claim in enumerate(claims):
            reorganized_evidence = reorganize_evidence(
                claim["claim"], claim["evidence"]
            )
            outputs.append(
                {"claim_id": claim["id"], "reorganized_evidence": reorganized_evidence}
            )
            pbar.update(1)

            if (i + 1) % 200 == 0:
                print(f"Save checkpoint at step {i+1}")
                with open(args.output_file, "w") as f:
                    f.write(json.dumps(outputs, indent=2))

    with open(args.output_file, "w") as f:
        f.write(json.dumps(outputs, indent=2))

import os
import json
import re
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator


import json
import torch
from modeling_fllama import FLlamaForCausalLM
from transformers import AutoConfig
import traceback

model_name = "meta-llama/Llama-3.2-1B-Instruct"
gsm8k = load_dataset("openai/gsm8k", "main")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
config.rope_scaling = None

model = FLlamaForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
    with_embedding_nodes=False
).cuda().eval()

print("Attention implementation:", model.config._attn_implementation)


with torch.no_grad():
    for layer in model.model.layers:
        layer.attn_write_log_alphas.data.fill_(20.0)
        layer.q_read_log_alphas.data.fill_(20.0)
        layer.k_read_log_alphas.data.fill_(20.0)
        layer.v_read_log_alphas.data.fill_(20.0)
        layer.mlp_write_log_alphas.data.fill_(20.0)
        layer.mlp_read_log_alphas.data.fill_(20.0)
    model.model.final_read_log_alphas.data.fill_(20.0)

model.set_edge_threshold_for_deterministic(0)
model.set_node_threshold_for_deterministic(0)

x = []
z = []
y = []

x.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
z.append("There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
y.append("6")

x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
y.append("5")

x.append("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
z.append("Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
y.append("39")

x.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
z.append("Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
y.append("8")

x.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
z.append("Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
y.append("9")

x.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
z.append("There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
y.append("29")

x.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
z.append("Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
y.append("33")

x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
z.append("Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
y.append("8")


examples = []
for i in range(len(x)):
    examples.append({
        'question': x[i],
        'answer': z[i]
    })

few_shot_examples = examples

def create_few_shot_prompt(examples, new_question):
    prompt = "Below are example math problems with their solutions:\n\n"
    for i, ex in enumerate(examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"Question: {ex['question']}\n"
        prompt += f"Answer: {ex['answer']}\n\n"

    prompt += "Now solve the following new problem:\n"
    prompt += f"Question: {new_question}\nAnswer:"

    return prompt

def improved_extract_numeric_answer(text):

    pattern = r'^####\s*(.*)$'
    lines = re.findall(pattern, text, flags=re.MULTILINE)
    if lines:
        final_line = lines[-1].strip()
        final_line_no_commas = final_line.replace(",", "")
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", final_line_no_commas)
        if nums:
            return nums[-1]

    text_no_commas = text.replace(",", "")
    all_nums = re.findall(r"[-+]?\d*\.\d+|\d+", text_no_commas)
    if all_nums:
        return all_nums[-1]
    return None


def generate_answer(model, tokenizer, prompt,
                    max_new_tokens=1024, temperature=0.6, top_p=0.9):
    prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
        )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text



def evaluate_model_on_gsm8k_single_question(
    model,
    tokenizer,
    gsm8k_dataset,
    few_shot_examples,
    num_test_samples=None,
    output_csv="gsm8k_results_single_question.csv"
):

    if num_test_samples is None:
        num_test_samples = len(gsm8k_dataset["test"])

    results = []
    for i in tqdm(range(num_test_samples), desc="Single-Question Inference"):
        item = gsm8k_dataset["test"][i]
        question = item["question"]
        gold_answer = item["answer"]
        prompt = create_few_shot_prompt(few_shot_examples, question)
        raw_answer = generate_answer(model, tokenizer, prompt)
        assistant_prefix = "Assistant:"
        if assistant_prefix in raw_answer:
            assistant_reply = raw_answer.split(assistant_prefix, 1)[1].strip()
        else:
            assistant_reply = raw_answer
        pred_number = improved_extract_numeric_answer(assistant_reply)
        gold_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", gold_answer)
        gold_number = gold_numbers[-1] if gold_numbers else None
        is_correct = (pred_number == gold_number) and (pred_number is not None)
        results.append({
            "Index": i,
            "Question": question,
            "Gold Answer": gold_answer,
            "Prompt": prompt,
            "Model Answer": raw_answer,
            "Predicted Number": pred_number,
            "Gold Number": gold_number,
            "Is Correct": is_correct
        })
    df = pd.DataFrame(results)
    accuracy = df["Is Correct"].mean()
    summary_row = {
        "Index": "Overall Accuracy",
        "Question": "",
        "Gold Answer": "",
        "Prompt": "",
        "Model Answer": "",
        "Predicted Number": "",
        "Gold Number": "",
        "Is Correct": accuracy
    }
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}. Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    evaluate_model_on_gsm8k_single_question(
        model=model,
        tokenizer=tokenizer,
        gsm8k_dataset=gsm8k,
        few_shot_examples=few_shot_examples,
        num_test_samples= None,
        output_csv="gsm8k_results_single_question.csv"
    )
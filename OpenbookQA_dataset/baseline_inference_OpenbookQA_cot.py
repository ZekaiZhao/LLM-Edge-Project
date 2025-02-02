import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoConfig

# If using your custom FLlamaForCausalLM:
from modeling_fllama import FLlamaForCausalLM
import random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

dataloader_num_workers=0
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

# Now continue importing the rest:
from transformers import set_seed
set_seed(SEED)

##################################################
# 1. LOAD YOUR TRAINED MODEL & TOKENIZER
##################################################
model_name = "meta-llama/Llama-3.2-1B-Instruct"


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



accelerator = Accelerator()
model, tokenizer = accelerator.prepare(model, tokenizer)


ds = load_dataset("allenai/openbookqa", "main")
openbookqa_test = ds["test"]

few_shot_examples = [
    {
        "id": "7-980",
        "question_stem": "The sun is responsible for",
        "choices": {
            "text": [
                "puppies learning new tricks",
                "children growing up and getting old",
                "flowers wilting in a vase",
                "plants sprouting, blooming and wilting"
            ],
            "label": ["A", "B", "C", "D"]
        },
        "ground_truth": "D",
        "Chain-of-thought": (
            "To determine the correct answer, let's analyze each option in relation to the sun's influence:\n\n"
            "A) Puppies learning new tricks: The sun does not directly influence the cognitive abilities or learning processes of puppies.\n\n"
            "B) Children growing up and getting old: While the sun provides essential energy and light that supports life on Earth, the process of growing and aging is primarily driven by biological factors.\n\n"
            "C) Flowers wilting in a vase: This is usually due to lack of water or cutting; not primarily the sun.\n\n"
            "D) Plants sprouting, blooming, and wilting: The sun's energy is crucial for photosynthesis and the entire life cycle of plants.\n"
            "Thus, the correct answer is D."
        )
    },
    {
        "id": "7-584",
        "question_stem": "When standing miles away from Mount Rushmore",
        "choices": {
            "text": [
                "the mountains seem very close",
                "the mountains are boring",
                "the mountains look the same as from up close",
                "the mountains seem smaller than in photographs"
            ],
            "label": ["A", "B", "C", "D"]
        },
        "ground_truth": "D",
        "Chain-of-thought": (
            "A) \"the mountains seem very close\" is incorrect; from miles away, they do not appear closer.\n"
            "B) \"the mountains are boring\" is subjective and not about visual perception.\n"
            "C) \"the mountains look the same as from up close\" is unlikely because distance changes perceived size.\n"
            "D) \"the mountains seem smaller than in photographs\" makes sense because photographs can be taken with zoom or perspective. The best answer is D."
        )
    },
    {
        "id": "7-870",
        "question_stem": "When food is reduced in the stomach",
        "choices": {
            "text": [
                "the mind needs time to digest",
                "take a second to digest what I said",
                "nutrients are being deconstructed",
                "reader's digest is a body of works"
            ],
            "label": ["A", "B", "C", "D"]
        },
        "ground_truth": "C",
        "Chain-of-thought": (
            "A) and B) refer to metaphorical digestion of information, not physical digestion.\n"
            "C) \"nutrients are being deconstructed\" is exactly what happens in digestion.\n"
            "D) is unrelated.\n"
            "Hence, correct answer is C."
        )
    },
    {
        "id": "7-321",
        "question_stem": "Stars are",
        "choices": {
            "text": [
                "warm lights that float",
                "made out of nitrate",
                "great balls of gas burning billions of miles away",
                "lights in the sky"
            ],
            "label": ["A", "B", "C", "D"]
        },
        "ground_truth": "C",
        "Chain-of-thought": (
            "A) Vague.\n"
            "B) Incorrect; not made of nitrate.\n"
            "C) Scientifically accurate, as they are balls of gas undergoing nuclear fusion, far away.\n"
            "D) Oversimplified.\n"
            "Correct answer is C."
        )
    },
    {
        "id": "9-732",
        "question_stem": "You can make a telescope with a",
        "choices": {
            "text": [
                "straw",
                "Glass",
                "Candle",
                "mailing tube"
            ],
            "label": ["A", "B", "C", "D"]
        },
        "ground_truth": "D",
        "Chain-of-thought": (
            "A) Straw is too flimsy.\n"
            "B) Glass is a material, not a tube.\n"
            "C) Candle is irrelevant.\n"
            "D) A mailing tube can serve as the sturdy cylinder to mount lenses.\n"
            "Hence, D."
        )
    },
]


def create_openbookqa_prompt(few_shot_data, new_question_stem, new_choices):

    prompt = "Below are example OpenbookQA problems, each with a chain-of-thought solution and final answer:\n\n"
    for i, ex in enumerate(few_shot_data):
        prompt += f"Example {i + 1}:\n"
        prompt += f"Q: {ex['question_stem']}\n"
        # List the choices:
        for choice_label, choice_text in zip(ex["choices"]["label"], ex["choices"]["text"]):
            prompt += f"  {choice_label}) {choice_text}\n"
        prompt += f"\nChain-of-thought:\n{ex['Chain-of-thought']}\n"
        prompt += f"Final Answer: {ex['ground_truth']}\n\n"

    prompt += "Now solve the following new problem:\n"
    prompt += f"Q: {new_question_stem}\n"
    for choice_label, choice_text in zip(new_choices["label"], new_choices["text"]):
        prompt += f"  {choice_label}) {choice_text}\n"
    prompt += "\nPlease show your reasoning step by step, and then give one final answer.\n"

    return prompt


################################################################################
# 4. Generate Function
################################################################################

def generate_answer(model, tokenizer, prompt,
                    max_new_tokens=1024, temperature=0.2, top_p=0.9):
    """
    Generates a single answer from the model for the given prompt,
    using a chat-style template if desired.
    """
    chat_prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

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



def extract_final_answer_from_text(text):

    pattern = r'(?i)final answer:\s*(.*)'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None



def evaluate_model_on_openbookqa(
        model,
        tokenizer,
        dataset_split,
        few_shot_data,
        num_test_samples=10,  # set to None or len(dataset_split) to run on entire set
        output_csv="openbookqa_results.csv"
):

    if num_test_samples is None or num_test_samples > len(dataset_split):
        num_test_samples = len(dataset_split)

    results = []

    for i in tqdm(range(num_test_samples), desc="OpenbookQA Inference"):
        item = dataset_split[i]


        question_stem = item["question_stem"]
        choices_text = item["choices"]["text"]
        choices_label = item["choices"]["label"]
        gold_label = item["answerKey"]


        prompt = create_openbookqa_prompt(few_shot_data, question_stem, {
            "text": choices_text,
            "label": choices_label
        })


        raw_answer = generate_answer(model, tokenizer, prompt)


        predicted_label = extract_final_answer_from_text(raw_answer)
        is_correct = (predicted_label == gold_label)

        results.append({
            "Index": i,
            "Question": question_stem,
            "Choices": choices_text,
            "Gold Label": gold_label,
            "Model Output": raw_answer,
            "Predicted Label": predicted_label,
            "Is Correct": is_correct
        })


    df = pd.DataFrame(results)
    accuracy = df["Is Correct"].mean()
    summary_row = {
        "Index": "Overall Accuracy",
        "Question": "",
        "Choices": "",
        "Gold Label": "",
        "Model Output": "",
        "Predicted Label": "",
        "Is Correct": accuracy
    }
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}. Accuracy: {accuracy:.2%}")



if __name__ == "__main__":


    openbookqa = load_dataset("openbookqa", "main")
    test_split = openbookqa["test"]


    evaluate_model_on_openbookqa(
        model=model,
        tokenizer=tokenizer,
        dataset_split=test_split,
        few_shot_data=few_shot_examples,
        num_test_samples= None,
        output_csv="openbookqa_results.csv"
    )

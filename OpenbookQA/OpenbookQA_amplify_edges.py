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
import torch.nn as nn

import numpy as np
import random
from transformers import set_seed
from modeling_fllama_soft import FLlamaForCausalLM
from transformers import AutoConfig

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###############################################################################
# Set random seeds for reproducibility
###############################################################################
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

###############################################################################
# Load dataset, tokenizer, and model
###############################################################################
model_name = "meta-llama/Llama-3.2-1B-Instruct"
#test_subset = gsm8k["test"].shuffle(seed=42).select(range(300))

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
config.rope_scaling = None
config.vocab_size = 128257  # match the checkpoint exactly

model_name_2 = "/home/zekai/zzk_project_llm1/GSM8k_run_Dec27th/inference/few_shot_inference_pruned_model_3.1b_2/consistency_verification/openbookqa_pruning_outputs_cot/checkpoint-800"

model = FLlamaForCausalLM.from_pretrained(
    model_name_2,
    config=config,
    torch_dtype=torch.bfloat16,
    with_embedding_nodes=False
).cuda().eval()


###############################################################################
# Helper functions for ranking, initializing, and setting top edges
###############################################################################
def load_ranked_edges(filepath):
    edges_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            parts = line.split()
            # Optionally skip lines that are too short
            if len(parts) < 5:
                continue

            try:
                layer_idx = int(parts[0])
                edge_type = parts[1]
                writer_idx = int(parts[2])

                # Some lines might have 'None' or '=====', so we handle that
                if parts[3].lower() == "none":
                    head_idx = None
                else:
                    head_idx = int(parts[3])

                alpha_val = float(parts[4])

                edges_data.append((layer_idx, edge_type, writer_idx, head_idx, alpha_val))
            except ValueError:
                # If a line doesn't parse properly, skip it
                continue

    # Sort descending by alpha_val
    edges_data.sort(key=lambda x: x[4], reverse=True)
    return edges_data

# Usage:
# edges_data = load_ranked_edges("ranked_edges_by_average_log_alpha.txt")
# Now `edges_data` will be a list of tuples like:
# [(layer_idx, edge_type, w_idx, h_idx, alpha), ...] sorted by alpha descending.


def init_all_ones_for_layer(layer, factor=1.0):
    layer.q_read_weights = nn.Parameter(torch.ones_like(layer.q_read_log_alphas) * factor)
    layer.k_read_weights = nn.Parameter(torch.ones_like(layer.k_read_log_alphas) * factor)
    layer.v_read_weights = nn.Parameter(torch.ones_like(layer.v_read_log_alphas) * factor)
    # layer.mlp_read_weights = nn.Parameter(torch.ones_like(layer.mlp_read_log_alphas) * factor)

def set_top_edges_factor(model, edges_data, top_k=100, new_factor=1.2):
    for layer_idx, edge_type, writer_idx, head_idx, _ in edges_data[:top_k]:
        layer = model.model.layers[layer_idx]
        if edge_type == "Q":
            layer.q_read_weights.data[writer_idx, head_idx] = new_factor
        elif edge_type == "K":
            layer.k_read_weights.data[writer_idx, head_idx] = new_factor
        elif edge_type == "V":
            layer.v_read_weights.data[writer_idx, head_idx] = new_factor
        # elif edge_type == "MLP":
        #     layer.mlp_read_weights.data[writer_idx] = new_factor

###############################################################################
# Few-shot examples, prompt creation, and single-evaluation function
###############################################################################
# inference

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
    """
    Build a few-shot prompt that includes chain-of-thought solutions in the examples,
    then asks the new question last.

    few_shot_data: list of dicts with keys:
       'question_stem', 'Chain-of-thought', 'ground_truth', 'choices'
    new_question_stem: str, the question text for the new item
    new_choices: dict like {"text": [...], "label": [...]}
    """
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
    # If your tokenizer has a built-in chat template, apply it; otherwise, just tokenize:
    # For demonstration, let's assume `tokenizer.apply_chat_template(...)` is valid.
    # If you don't have that, just do the plain prompt.

    # Convert user prompt into the "Assistant" style if needed
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


################################################################################
# 5. Simple Extraction of Final Answer (A/B/C/D)
################################################################################

def extract_final_answer_from_text(text):
    """
    Find all occurrences of 'Final Answer:' (case-insensitive)
    and return whatever follows it on the same line, from the last match.
    """
    pattern = r'(?i)final answer:\s*(.*)'
    matches = re.findall(pattern, text)
    if matches:
        # Return the last occurrence
        return matches[-1].strip()
    return None


################################################################################
# 6. Evaluation on OpenbookQA
################################################################################

def evaluate_model_on_openbookqa(
    model,
    tokenizer,
    dataset_split,
    few_shot_data,
    num_test_samples=None
):
    """
    Evaluate the model in a few-shot manner on OpenbookQA and return accuracy only.
    """
    if num_test_samples is None or num_test_samples > len(dataset_split):
        num_test_samples = len(dataset_split)

    correct_count = 0

    for i in range(num_test_samples):
        item = dataset_split[i]

        # Extract question, choices, gold
        question_stem = item["question_stem"]
        choices_text = item["choices"]["text"]
        choices_label = item["choices"]["label"]
        gold_label = item["answerKey"]  # The correct label (A, B, C, D)

        # Create a few-shot prompt
        prompt = create_openbookqa_prompt(
            few_shot_data,
            question_stem,
            {
                "text": choices_text,
                "label": choices_label
            }
        )

        # Generate a raw answer
        raw_answer = generate_answer(model, tokenizer, prompt)

        # Extract final A/B/C/D from the raw text
        predicted_label = extract_final_answer_from_text(raw_answer)

        # Check correctness
        if predicted_label == gold_label:
            correct_count += 1

    accuracy = correct_count / num_test_samples
    return accuracy


################################################################################
# 7. Main (example usage)
################################################################################
###############################################################################
# Main: Automatic search over new_factor in [1.2, 1.3] with step 0.1
###############################################################################
if __name__ == "__main__":
    # Pre-rank edges once (the log alphas won't change)
    edges_data = load_ranked_edges("/home/zekai/zzk_project_llm1/GSM8k_run_Dec27th/inference/few_shot_inference_pruned_model_3.1b_2/Amplify_edges/intersection_edges.txt")

    candidate_factors = np.arange(1.4, 1.6, 0.05)  # add a small epsilon so that 1.3 is included
    candidate_factors = [round(x, 2) for x in candidate_factors]
    # If you want smaller increments, you could do:
    # import numpy as np
    # candidate_factors = np.arange(1.2, 1.31, 0.1)  # [1.2, 1.3] stepping by 0.1

    best_factor = None
    best_accuracy = -1.0

    all_results = []  # will hold tuples of (factor, accuracy)

    for factor in candidate_factors:
        # 1) Reset all read weights to 1.0
        for layer in model.model.layers:
            init_all_ones_for_layer(layer, factor=1.0)

        # 2) Set top-K edges to the new factor
        set_top_edges_factor(model, edges_data, top_k=150438, new_factor=factor)

        ###############################################################################
        # Initialize the logs to large positive so that no edges/heads are pruned
        ###############################################################################
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

        # 3) Evaluate the model
        accuracy = evaluate_model_on_openbookqa(
            model,
            tokenizer,
            openbookqa_test,
            few_shot_examples,
            num_test_samples=None
        )
        print(f"new_factor = {factor}, accuracy = {accuracy:.3f}")

        # 4) Store the result
        all_results.append((factor, accuracy))

    # Sort all_results by accuracy descending
    all_results.sort(key=lambda x: x[1], reverse=True)

    # Now pick the top 5
    best_5 = all_results[:5]

    print("\n OpenbookQA Seed = 42 N= 150438, Top 5 factors by accuracy:")
    for i, (factor, accuracy) in enumerate(best_5, start=1):
        print(f"Rank {i}: factor = {factor}, accuracy = {accuracy:.3f}")


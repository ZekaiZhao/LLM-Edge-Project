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
model_path = "/home/zekai/zzk_project_llm1/GSM8k_run_Dec27th/inference/few_shot_inference_pruned_model_3.1b_2/consistency_verification/outputs_27/checkpoint-1000"  # Path to your finetuned LogiQA model


tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
config.rope_scaling = None
config.vocab_size = 128257  # match the checkpoint exactly

model = FLlamaForCausalLM.from_pretrained(
    model_path,
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


# Accelerator (optional for mixed precision, multi-GPU, etc.)
accelerator = Accelerator()
model, tokenizer = accelerator.prepare(model, tokenizer)


def create_few_shot_examples_from_train(train_split, n=3):
    """
    Extract `n` examples from the OpenBookQA train split.
    Returns a list of dicts where each dict has:
      - "question": str
      - "choices": {"A": ..., "B": ..., "C": ..., "D": ...}
      - "answer": one of {"A","B","C","D"}
      - "reasoning": optional string (if you want a chain-of-thought)
    """
    # Shuffle and pick n examples or just pick the first n
    indices = list(range(len(train_split)))
    random.shuffle(indices)
    chosen_indices = indices[:n]

    examples = []
    for i in chosen_indices:
        item = train_split[i]
        question = item["question_stem"]
        # item["choices"]["label"] might be ["A","B","C","D"]
        # item["choices"]["text"] might be corresponding text
        choices_dict = dict(zip(item["choices"]["label"], item["choices"]["text"]))
        gold_answer = item["answerKey"]

        # Optionally grab the 'fact1' field if you want a short explanation:
        # Some items have item["fact1"] = "some short fact"
        # We'll store it as "reasoning" just to show how you'd do it
        reasoning_text = item.get("fact1", "")

        ex = {
            "question": question,
            "choices": choices_dict,
            "answer": gold_answer,
            "reasoning": reasoning_text,   # optional
        }
        examples.append(ex)
    return examples

def create_few_shot_prompt_openbookqa(examples, new_question, new_choices):
    """
    Builds a chat-like prompt for OpenBookQA-style multiple choice questions.

    `examples` is a list of dicts, each with:
      - "question": str
      - "choices": dict like {"A": "...", "B": "...", "C": "...", "D": "..."}
      - "answer": one of {"A","B","C","D"}
      - "reasoning": optional string (explanation)

    `new_question` is the question we want the model to solve
    `new_choices` is a dict with A/B/C/D answer choices

    Returns a string that shows those few-shot Q&A examples, then the new question.
    """
    prompt = "Below are example multiple-choice questions with their correct answers:\n\n"
    for i, ex in enumerate(examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"Question: {ex['question']}\n"
        prompt += "Choices:\n"
        for letter, choice_text in ex['choices'].items():
            prompt += f"  {letter}) {choice_text}\n"
        if 'reasoning' in ex and len(ex['reasoning']) > 0:
            prompt += f"Explanation: {ex['reasoning']}\n"
        prompt += f"Correct Answer: {ex['answer']}\n\n"

    # Now add the new question
    prompt += "Now please answer the new question:\n"
    prompt += f"Question: {new_question}\n"
    prompt += "Choices:\n"
    for letter, choice_text in new_choices.items():
        prompt += f"  {letter}) {choice_text}\n"
    prompt += "\nAnswer:"
    return prompt


def generate_answer(model, tokenizer, prompt,
                    max_new_tokens=1024, temperature=0.6, top_p=0.9):
    """
    Generates a single answer from the model for the given prompt,
    using a chat-style format.
    """
    # Wrap user prompt in chat template
    chat_input = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
    )

    inputs = tokenizer(chat_input, return_tensors="pt", padding=True).to(model.device)

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



def extract_openbookqa_choice(text):
    """
    Searches the entire output for lines like:
       - "Answer: C"
       - "Correct Answer: C"
       - "The correct answer is C"
    and returns the *last* such letter found (A, B, C, or D).
    If none is found, returns None.
    """
    text_upper = text.upper()

    # 1) Look for lines like "Answer: B", "Correct answer: C", "The correct answer is D", etc.
    #    This pattern allows optional punctuation like ":" or "-" or "=" and optional parentheses.
    #    Example matches: "ANSWER: B", "THE CORRECT ANSWER IS B)", etc.
    pattern_labeled = r"(?:THE CORRECT ANSWER IS|ANSWER|CORRECT ANSWER)\s*[:=\-]?\s*\(?([ABCD])\)?"
    labeled_matches = re.findall(pattern_labeled, text_upper)

    if labeled_matches:
        # If multiple, take the last occurrence
        return labeled_matches[-1]

    # 2) If not found, fallback: look for a standalone letter A/B/C/D near the end
    pattern_standalone = r"\b([ABCD])\b"
    standalone_matches = re.findall(pattern_standalone, text_upper)
    if standalone_matches:
        # Take the last match
        return standalone_matches[-1]
    # The last match in the list is presumably the model's final pick

    return None

def evaluate_model_on_openbookqa_single_question(
    model,
    tokenizer,
    train_split,
    test_split,
    num_few_shot=5,
    output_csv="openbookqa_results.csv",
    num_test_samples=None
):
    """
    Evaluates the model on OpenBookQA in a single-question manner, using
    a few-shot prompt from the training set. Optionally limit the test set
    to `num_test_samples` items.
    """
    # Gather your few-shot examples from the training set.
    few_shot_examples = create_few_shot_examples_from_train(train_split, n=num_few_shot)

    # If the user specifies num_test_samples, limit the test set
    if num_test_samples is not None:
        test_indices = list(range(min(num_test_samples, len(test_split))))
    else:
        test_indices = list(range(len(test_split)))

    results = []
    for i in tqdm(test_indices, desc="OpenBookQA Inference"):
        item = test_split[i]
        question = item["question_stem"]
        # Convert "label":["A","B","C","D"], "text":[...,...,...,...] to dict
        choices_dict = dict(zip(item["choices"]["label"], item["choices"]["text"]))
        gold_answer = item["answerKey"]  # e.g. "A","B","C","D"

        # Build the prompt from the few-shot examples
        prompt = create_few_shot_prompt_openbookqa(
            examples=few_shot_examples,
            new_question=question,
            new_choices=choices_dict
        )

        # Generate the model's text output
        raw_answer = generate_answer(model, tokenizer, prompt)

        # Extract the chosen letter (A,B,C,D) from the text
        predicted_letter = extract_openbookqa_choice(raw_answer)

        # Check correctness
        is_correct = (predicted_letter == gold_answer)

        results.append({
            "Index": i,
            "Question": question,
            "Choices": choices_dict,
            "Gold Answer": gold_answer,
            "Model Output": raw_answer,
            "Predicted Letter": predicted_letter,
            "Is Correct": is_correct
        })

    # Convert to DataFrame & measure accuracy
    df = pd.DataFrame(results)
    accuracy = df["Is Correct"].mean()
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}. Accuracy: {accuracy:.2%}")
    return df


if __name__ == "__main__":
    openbookqa = load_dataset("allenai/openbookqa", "main")
    train_split = openbookqa["train"]
    test_split = openbookqa["test"]

    evaluate_model_on_openbookqa_single_question(
        model=model,
        tokenizer=tokenizer,
        train_split=train_split,
        test_split=test_split,
        num_few_shot=5,  # how many training examples to include in the prompt
        num_test_samples=None,  # only test on 50 items
        output_csv="openbookqa_results_subset.csv"
    )


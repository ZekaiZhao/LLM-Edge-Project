import logging
import sys
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

# Import the pruning-enabled model and trainer from your snippet
from modeling_fllama import FLlamaForCausalLM
from fllama_boolean_expressions_fs import FLlamaTrainer
#from your_data_loading_script import load_datasets, DataCollatorBool  # Adjust as needed

logger = logging.getLogger(__name__)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import json
import math
import copy
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import re

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

from transformers import AutoConfig
import random

# For complete reproducibility:
SEED = 21
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

# model_name_or_path was previously defined and should be left unchanged
model_name_or_path = 'meta-llama/Llama-3.2-1B-Instruct'

# Load the tokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Ensure the tokenizer has a pad token before creating the DataLoader
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

# Get the config first
config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
config.rope_scaling = None
# Ensure 'rope_scaling' is in the config and has a 'type' key
# Use a default 'type' if not present
#config.rope_scaling = config.rope_scaling or {}  # Create if not exists
#config.rope_scaling["type"] = config.rope_scaling.get("type", "linear")  # Set to 'linear' if not found

# Load the model from a pretrained checkpoint or a local directory
model = FLlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",  # Example model name or local path
    config=config,
    with_embedding_nodes=False,          # Adjust based on your setup
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False

print("Edge Sparsity:", model.get_edge_sparsity())
print("Node Sparsity:", model.get_node_sparsity())

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
model.resize_token_embeddings(len(tokenizer))



def perturb_question(question: str) -> str:
    """
    Slightly shuffle or otherwise modify the question
    for generating the 'corr_input_ids'.
    """
    words = question.split()
    if len(words) > 1:
        random.shuffle(words)
    return " ".join(words)


def preprocess_openbookqa(example, tokenizer, max_length=128):
    """
    1) Retrieves the question from 'question_stem'.
    2) Extracts the correct answer choice from 'answerKey'.
    3) Tokenizes question & correct answer, combines them.
    4) Creates 'labels' where question tokens are -100,
       answer tokens are the real IDs.
    5) Creates a 'corr_input_ids' from a perturbed question + correct answer.
    6) Returns a dictionary with keys:
       'input_ids', 'attention_mask', 'labels', 'corr_input_ids', and 'idxes'.
    """

    # 1. Extract question
    question = example["question_stem"]

    # 2. Identify the correct choice
    #    'answerKey' is one of {"A","B","C","D"},
    #    'choices["label"]' is like ["A","B","C","D"]
    correct_label = example["answerKey"]
    correct_index = example["choices"]["label"].index(correct_label)
    correct_answer = example["choices"]["text"][correct_index]

    # 3. Tokenize question and correct answer (no special tokens yet)
    question_enc = tokenizer(
        question,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )
    answer_enc = tokenizer(
        correct_answer,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )

    # 4. Combine question + answer into one sequence
    #    If your tokenizer has a special EOS token or separator, you can insert it.
    #    Below we just do question + answer.
    input_ids = question_enc["input_ids"] + answer_enc["input_ids"]
    attention_mask = [1] * len(input_ids)

    # For labels:
    #  - The question portion gets -100 (ignored by the loss).
    #  - The answer portion is the real token IDs.
    labels = (
        [-100] * len(question_enc["input_ids"])
        + answer_enc["input_ids"]
    )
    # 'idxes' is the start index of the answer portion
    idxes = len(question_enc["input_ids"])

    # 5. Generate corrupted question
    corrupted_question = perturb_question(question)
    corr_question_enc = tokenizer(
        corrupted_question,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )
    corr_input_ids = corr_question_enc["input_ids"] + answer_enc["input_ids"]

    # 6. Truncate if necessary
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        # Adjust idxes if it goes out of bounds
        if idxes >= max_length:
            idxes = max_length - 1

    if len(corr_input_ids) > max_length:
        corr_input_ids = corr_input_ids[:max_length]

    # If desired, you can also filter out IDs >= tokenizer.vocab_size:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        input_ids = [tid for tid in input_ids if tid < vocab_size]
        labels = [
            lab if (lab < vocab_size and lab != -100) else -100
            for lab in labels
        ]
        corr_input_ids = [tid for tid in corr_input_ids if tid < vocab_size]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "corr_input_ids": corr_input_ids,
        "idxes": idxes
    }


import torch
from torch.nn.utils.rnn import pad_sequence

def data_collator(features):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

    # Handle missing corr_input_ids
    corr_input_ids = [
        torch.tensor(
            f.get("corr_input_ids", [tokenizer.pad_token_id]),
            dtype=torch.long
        )
        for f in features
    ]

    # Default idx to 0 if missing
    idxes = [
        torch.tensor(f.get("idxes", 0), dtype=torch.long)
        for f in features
    ]

    # Pad sequences
    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0
    )
    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )
    corr_input_ids = pad_sequence(
        corr_input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    idxes = torch.stack(idxes, dim=0)  # no padding needed

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "corr_input_ids": corr_input_ids,
        "idxes": idxes
    }


# 1. Load dataset
dataset = load_dataset("allenai/openbookqa", "main")

# 2. Preprocess splits
max_length = 128

train_data = dataset["train"].map(
    lambda x: preprocess_openbookqa(x, tokenizer=tokenizer, max_length=max_length),
    batched=False
)
train_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels", "corr_input_ids", "idxes"]
)

val_data = dataset["validation"].map(
    lambda x: preprocess_openbookqa(x, tokenizer=tokenizer, max_length=max_length),
    batched=False
)
val_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels", "corr_input_ids", "idxes"]
)

test_data = dataset["test"].map(
    lambda x: preprocess_openbookqa(x, tokenizer=tokenizer, max_length=max_length),
    batched=False
)
test_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels", "corr_input_ids", "idxes"]
)

# 3. Create data loaders
train_loader = DataLoader(
    train_data,
    batch_size=1,
    shuffle=True,
    collate_fn=data_collator
)
val_loader = DataLoader(
    val_data,
    batch_size=1,
    shuffle=False,
    collate_fn=data_collator
)
test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False,
    collate_fn=data_collator
)

from transformers.utils import logging
logging.set_verbosity_info()

from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs_29",
    overwrite_output_dir=True,
    num_train_epochs= 10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps= 1,
    save_steps=500, #1000
    evaluation_strategy="steps",
    eval_steps=500, #1000
    load_best_model_at_end=True,
    disable_tqdm=False,
    gradient_checkpointing=True,
    seed=21  # Explicitly set the seed
)

target_edge_sparsity = 0.1
target_node_sparsity = 0.1

# Get the config first
config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
config.rope_scaling = None
# Ensure 'rope_scaling' is in the config and has a 'type' key
# Use a default 'type' if not present
#config.rope_scaling = config.rope_scaling or {}  # Create if not exists
#config.rope_scaling["type"] = config.rope_scaling.get("type", "linear")  # Set to 'linear' if not found

# Load the model from a pretrained checkpoint or a local directory
llama_model = FLlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",  # Example model name or local path
    config=config,
    with_embedding_nodes=False,
    torch_dtype=torch.bfloat16
)

print("Edge Sparsity at init:", llama_model.get_edge_sparsity())
print("Node Sparsity at init:", llama_model.get_node_sparsity())


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
llama_model.resize_token_embeddings(len(tokenizer))

# Correct usage:
from accelerate import Accelerator

# Initialize once at the very beginning
accelerator = Accelerator(mixed_precision="bf16")

from transformers import TrainerCallback
# Pass the same accelerator instance to FLlamaTrainer
trainer = FLlamaTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    start_edge_sparsity=0.0,
    target_edge_sparsity=target_edge_sparsity,
    start_node_sparsity=0.0,
    target_node_sparsity=target_node_sparsity,
    skip_node_loss_if_higher_sparsity=False,
    num_sparsity_warmup_steps=200, # 200
    warmup_type="linear",
    edges_lr= 0.8,    # 0.8
    nodes_lr= 0.8,       # 0.8
    reg_edges_lr=0.4, # 0.4
    reg_nodes_lr= 0.4,    # 0.4
    warmup_steps=200, # needed 200
    disable_node_loss=False,
    llama_model=llama_model
)

class SparsityCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Access trainer safely
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            print("[SparsityCallback Warning] Trainer instance not found in kwargs.")
            return control  # Return the unmodified control object
        model = trainer.model
        edge_s = model.get_edge_sparsity()
        node_s = model.get_node_sparsity()
        print(f"[Step {state.global_step}] Edge: {edge_s:.4f}, Node: {node_s:.4f}")

trainer.add_callback(SparsityCallback())

trainer.train()
final_edge_sparsity = model.get_edge_sparsity()
final_node_sparsity = model.get_node_sparsity()
print("Edge Sparsity after training:", final_edge_sparsity)
print("Node Sparsity after training:", final_node_sparsity)

metrics = trainer.evaluate()
print("Eval metrics:", metrics)

edge_s_final = model.get_edge_sparsity()
node_s_final = model.get_node_sparsity()
print("Final Edge Sparsity:", edge_s_final)
print("Final Node Sparsity:", node_s_final)


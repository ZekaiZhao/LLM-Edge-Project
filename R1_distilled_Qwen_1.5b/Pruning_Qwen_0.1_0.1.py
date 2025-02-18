

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


from modeling_fqwen import FQwenForCausalLM
from fqwen_boolean_expression_fs import FQwenTrainer

logger = logging.getLogger(__name__)

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


SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

dataloader_num_workers = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from transformers import set_seed
set_seed(SEED)


model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path
)

# Ensure a pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})


def data_collator(features):
    if isinstance(features, dict):
        batch_size = len(next(iter(features.values())))
        features = [
            {key: value[i] for key, value in features.items()}
            for i in range(batch_size)
        ]


    for f in features:
        if "attention_mask" not in f:

            f["attention_mask"] = [1] * len(f["input_ids"])


    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) if not torch.is_tensor(f["input_ids"]) else f["input_ids"]
                 for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) if not torch.is_tensor(f["attention_mask"]) else f["attention_mask"]
                      for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) if not torch.is_tensor(f["labels"]) else f["labels"]
              for f in features]
    corr_input_ids = [
        torch.tensor(f.get("corr_input_ids", [tokenizer.pad_token_id]), dtype=torch.long)
        if not torch.is_tensor(f.get("corr_input_ids", [tokenizer.pad_token_id])) else f.get("corr_input_ids", [tokenizer.pad_token_id])
        for f in features
    ]
    idxes = [torch.tensor(f.get("idxes", 0), dtype=torch.long) if not torch.is_tensor(f.get("idxes", 0)) else f.get("idxes", 0)
             for f in features]


    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    corr_input_ids = pad_sequence(corr_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    idxes = torch.stack(idxes, dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "corr_input_ids": corr_input_ids,
        "idxes": idxes
    }


def perturb_question(question):
    words = question.split()
    if len(words) > 1:
        random.shuffle(words)
    return " ".join(words)

def preprocess_gsm8k(example, tokenizer, max_length=128):

    question = example["question"]
    answer = example["answer"]


    question_enc = tokenizer(
        question,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )
    answer_enc = tokenizer(
        answer,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )


    input_ids = question_enc["input_ids"] + answer_enc["input_ids"]

    attention_mask = [1] * len(input_ids)


    labels = [-100] * len(question_enc["input_ids"]) + answer_enc["input_ids"]


    input_ids = input_ids[:max_length]
    labels = labels[:max_length]



    corrupted_question = perturb_question(question)
    corr_question_enc = tokenizer(
        corrupted_question,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )
    corr_input_ids = corr_question_enc["input_ids"] + answer_enc["input_ids"]

    corr_input_ids = corr_input_ids[:max_length]


    idxes = min(len(question_enc["input_ids"]), max_length - 1)


    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        if idxes >= max_length:
            idxes = max_length - 1
    if len(corr_input_ids) > max_length:
        corr_input_ids = corr_input_ids[:max_length]


    input_ids = [tid for tid in input_ids if tid < tokenizer.vocab_size]
    labels = [lid if lid < tokenizer.vocab_size else -100 for lid in labels]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "corr_input_ids": corr_input_ids,
        "idxes": idxes
    }



from datasets import load_dataset
gsm8k = load_dataset("openai/gsm8k", "main")
max_length = 128


train_data = gsm8k["train"].map(
    lambda x: preprocess_gsm8k(x, tokenizer=tokenizer, max_length=max_length),
    batched=False
)

train_data.set_format(type="torch", columns=["input_ids","attention_mask","labels","corr_input_ids","idxes"])

test_data = gsm8k["test"].map(
    lambda x: preprocess_gsm8k(x, tokenizer=tokenizer, max_length=max_length),
    batched=False
)
test_data.set_format(type="torch", columns=["input_ids","attention_mask","labels","corr_input_ids","idxes"])


train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=data_collator)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=data_collator)


from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs_qwen_gsm8k_2",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps=1,
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    disable_tqdm=False,
    gradient_checkpointing=True,
    seed=42
)

target_edge_sparsity = 0.1
target_node_sparsity = 0.1


from transformers import AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)

model = FQwenForCausalLM.from_pretrained(
    model_name_or_path,
    config=config,
    with_embedding_nodes=False,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.config.use_cache = False

logger.info(f"Edge Sparsity: {model.get_edge_sparsity()}")
logger.info(f"Node Sparsity: {model.get_node_sparsity()}")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
model.resize_token_embeddings(len(tokenizer))

model.train()


from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="bf16")

training_args.dataloader_num_workers = 0

from transformers import TrainerCallback
trainer = FQwenTrainer(
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
    num_sparsity_warmup_steps=200,
    warmup_type="linear",
    edges_lr=0.8,
    nodes_lr=0.8,
    reg_edges_lr=0.4,
    reg_nodes_lr=0.4,
    warmup_steps=200,
    disable_node_loss=False,
    ref_model=None
)

class SparsityCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            print("[SparsityCallback Warning] Trainer not found.")
            return control
        model = trainer.model
        edge_s = model.get_edge_sparsity()
        node_s = model.get_node_sparsity()
        print(f"[Step {state.global_step}] Edge: {edge_s:.4f}, Node: {node_s:.4f}")
        return control

trainer.add_callback(SparsityCallback())


trainer.train()

final_edge_sparsity = model.get_edge_sparsity()
final_node_sparsity = model.get_node_sparsity()
print("Edge Sparsity after training:", final_edge_sparsity)
print("Node Sparsity after training:", final_node_sparsity)

metrics = trainer.evaluate()
print("Eval metrics:", metrics)

print("Final Edge Sparsity:", model.get_edge_sparsity())
print("Final Node Sparsity:", model.get_node_sparsity())
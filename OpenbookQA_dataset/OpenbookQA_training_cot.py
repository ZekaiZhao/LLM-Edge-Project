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


from modeling_fllama import FLlamaForCausalLM
from fllama_boolean_expressions_fs import FLlamaTrainer

import numpy as np
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset


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


from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_info()


model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"


from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})


def data_collator(features):

    input_ids = [f["input_ids"].clone().detach().long() for f in features]
    attention_mask = [f["attention_mask"].clone().detach().long() for f in features]
    labels = [f["labels"].clone().detach().long() for f in features]

    corr_input_ids = [torch.tensor(f.get("corr_input_ids", [tokenizer.pad_token_id]), dtype=torch.long)
                      for f in features]

    idxes = [torch.tensor(f.get("idxes", 0), dtype=torch.long) for f in features]


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


dataset = load_dataset("json", data_files={
    "train": "train_openbookQA_cot.json",
    "test": "test_openbookQA_cot.json"
})


def perturb_question(question):

    words = question.split()
    if len(words) > 1:
        random.shuffle(words)
    return " ".join(words)

def preprocess_multiple_choice(example, tokenizer, max_length=128):

    question = example["question_stem"]


    all_labels = example["choices"]["label"]
    all_texts = example["choices"]["text"]
    correct_label = example["ground_truth"]


    label_to_text = dict(zip(all_labels, all_texts))
    if correct_label not in label_to_text:
        correct_answer = ""
    else:
        correct_answer = label_to_text[correct_label]


    chain_of_thought = example.get("Chain-of-thought", "")


    question_enc = tokenizer(
        question,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )
    answer_enc = tokenizer(
        chain_of_thought + correct_answer,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )


    input_ids = question_enc["input_ids"] + answer_enc["input_ids"]
    attention_mask = [1] * len(input_ids)


    labels = [-100] * len(question_enc["input_ids"]) + answer_enc["input_ids"]


    idxes = len(question_enc["input_ids"])


    corrupted_question = perturb_question(question)
    corr_question_enc = tokenizer(
        corrupted_question,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )
    corr_input_ids = corr_question_enc["input_ids"] + answer_enc["input_ids"]


    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        if idxes >= max_length:
            idxes = max_length - 1

    if len(corr_input_ids) > max_length:
        corr_input_ids = corr_input_ids[:max_length]


    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    input_ids = [tid for tid in input_ids if tid < vocab_size]
    labels = [lid if (lid < vocab_size and lid != -100) else -100 for lid in labels]
    corr_input_ids = [cid for cid in corr_input_ids if cid < vocab_size]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "corr_input_ids": corr_input_ids,
        "idxes": idxes
    }


max_length = 128
train_data = dataset["train"].map(
    lambda x: preprocess_multiple_choice(x, tokenizer=tokenizer, max_length=max_length),
    batched=False
)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "corr_input_ids", "idxes"])

test_data = dataset["test"].map(
    lambda x: preprocess_multiple_choice(x, tokenizer=tokenizer, max_length=max_length),
    batched=False
)
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "corr_input_ids", "idxes"])

# Create PyTorch DataLoaders
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=data_collator)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=data_collator)


config = AutoConfig.from_pretrained(model_name_or_path)

config.rope_scaling = None

model = FLlamaForCausalLM.from_pretrained(
    model_name_or_path,
    config=config,
    with_embedding_nodes=False,
    torch_dtype=torch.bfloat16
)
model.config.use_cache = False


print("Edge Sparsity at init:", model.get_edge_sparsity())
print("Node Sparsity at init:", model.get_node_sparsity())


model.resize_token_embeddings(len(tokenizer))


llama_model = FLlamaForCausalLM.from_pretrained(
    model_name_or_path,
    config=config,
    with_embedding_nodes=False,
    torch_dtype=torch.bfloat16
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
llama_model.resize_token_embeddings(len(tokenizer))


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./openbookqa_pruning_outputs_cot_2",
    overwrite_output_dir=True,
    num_train_epochs=10,
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
    seed=SEED,
)

target_edge_sparsity = 0.1
target_node_sparsity = 0.1


from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="bf16")

from transformers import TrainerCallback

class SparsityCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        trainer_local = kwargs.get("trainer", None)
        if trainer_local is None:
            return control
        edge_s = trainer_local.model.get_edge_sparsity()
        node_s = trainer_local.model.get_node_sparsity()
        print(f"[Step {state.global_step}] Edge Sparsity: {edge_s:.4f}, Node Sparsity: {node_s:.4f}")
        return control

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
    num_sparsity_warmup_steps=200,
    warmup_type="linear",
    edges_lr=0.8,
    nodes_lr=0.8,
    reg_edges_lr=0.4,
    reg_nodes_lr=0.4,
    warmup_steps=200,
    disable_node_loss=False,
    llama_model=llama_model,
)

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
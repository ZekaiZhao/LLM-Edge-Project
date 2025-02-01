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
import random
import numpy as np

import json
import torch
from modeling_fllama_soft import FLlamaForCausalLM
from transformers import AutoConfig
import traceback
from pprint import pprint



model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})


config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
config.rope_scaling = None
config.vocab_size = 128257


model_name_2 = "Name to your model"


model = FLlamaForCausalLM.from_pretrained(
    model_name_2,
    config=config,
    torch_dtype=torch.bfloat16,
    with_embedding_nodes=False
).cuda().eval()


def print_all_edge_log_alphas(model):
    edges_data = rank_edges_across_all_layers(model)

    print("\n===== All edges by log_alpha (descending) =====")
    for i, (layer_idx, edge_type, writer_idx, head_idx, alpha_val) in enumerate(edges_data, start=1):
        head_str = f"Head={head_idx}" if head_idx is not None else "Head=None"
        print(f"{i:3d}) Layer={layer_idx}, {edge_type}-read, Writer={writer_idx}, {head_str}, log_alpha={alpha_val:.4f}")


def rank_edges_across_all_layers(model):
    edges_data = []


    for layer_idx, layer in enumerate(model.model.layers):
        q_alphas = layer.q_read_log_alphas.detach().cpu()
        num_writers_q, num_heads_q = q_alphas.shape
        for w_idx in range(num_writers_q):
            for h_idx in range(num_heads_q):
                alpha_val = float(q_alphas[w_idx, h_idx])
                edges_data.append((layer_idx, "Q", w_idx, h_idx, alpha_val))

        k_alphas = layer.k_read_log_alphas.detach().cpu()
        num_writers_k, num_kv_heads_k = k_alphas.shape
        for w_idx in range(num_writers_k):
            for h_idx in range(num_kv_heads_k):
                alpha_val = float(k_alphas[w_idx, h_idx])
                edges_data.append((layer_idx, "K", w_idx, h_idx, alpha_val))

        v_alphas = layer.v_read_log_alphas.detach().cpu()
        num_writers_v, num_kv_heads_v = v_alphas.shape
        for w_idx in range(num_writers_v):
            for h_idx in range(num_kv_heads_v):
                alpha_val = float(v_alphas[w_idx, h_idx])
                edges_data.append((layer_idx, "V", w_idx, h_idx, alpha_val))

    edges_data.sort(key=lambda x: x[4], reverse=True)

    return edges_data

edges_data = rank_edges_across_all_layers(model)
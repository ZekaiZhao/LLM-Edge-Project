# fqwen_boolen_expression_fs.py

import logging
import os
import sys
import math
import warnings
import torch
import random
import json

from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, load_from_disk, DatasetDict

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch.nn as nn
from torch.optim import AdamW

from accelerate import Accelerator

logger = logging.getLogger(__name__)


sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/modeling/"
    )
)

from modeling_fqwen import FQwenForCausalLM


class FQwenTrainer(Seq2SeqTrainer):
    """
    A specialized trainer that:
      - Sets up gating or sparsity constraints
      - Calls a "reference model" to compute KL or alignment losses
      - Merges the gating loss with the usual training loss
    """

    def __init__(self, *args, **kwargs):
        self.target_edge_sparsity = kwargs.pop('target_edge_sparsity', 0.0)
        self.start_edge_sparsity = kwargs.pop('start_edge_sparsity', 0.0)
        self.target_node_sparsity = kwargs.pop('target_node_sparsity', 0.0)
        self.start_node_sparsity = kwargs.pop('start_node_sparsity', 0.0)

        self.edges_lr = kwargs.pop('edges_lr', 0.8)
        self.nodes_lr = kwargs.pop('nodes_lr', 0.8)
        self.reg_edges_lr = kwargs.pop('reg_edges_lr', 0.8)
        self.reg_nodes_lr = kwargs.pop('reg_nodes_lr', 0.8)
        self.warmup_steps = kwargs.pop('warmup_steps', 0)
        self.disable_node_loss = kwargs.pop('disable_node_loss', False)

        # For warmup scheduling of sparsity:
        if "num_edge_sparsity_warmup_steps" in kwargs:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_edge_sparsity_warmup_steps')
        else:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', 0)
        if "num_node_sparsity_warmup_steps" in kwargs:
            self.num_node_sparsity_warmup_steps = kwargs.pop('num_node_sparsity_warmup_steps')
        else:
            self.num_node_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', self.num_edge_sparsity_warmup_steps)

        self.warmup_type = kwargs.pop('warmup_type', 'linear')
        # A separate reference QWen or baseline model:
        self.ref_model = kwargs.pop('ref_model', None)
        self.skip_node_loss_if_higher_sparsity = kwargs.pop('skip_node_loss_if_higher_sparsity', False)

        super().__init__(*args, **kwargs)

        # We can reset gating parameters:
        self.model.reset_all_log_alphas()
        if self.ref_model is not None:
            self.ref_model.reset_all_log_alphas()

        # Prepare models with Accelerator if needed
        self.accelerator = Accelerator(mixed_precision="bf16")
        self.model = self.accelerator.prepare(self.model)
        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare(self.ref_model)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer, self.lr_scheduler = get_optimizers(
            self.model,
            self.edges_lr,
            self.nodes_lr,
            self.reg_edges_lr,
            self.reg_nodes_lr,
            num_training_steps,
            warmup_steps=self.warmup_steps
        )

    def get_current_edge_target_sparsity(self, global_step):
        """
        Linearly or logarithmically ramp up the target edge sparsity
        from start_edge_sparsity to target_edge_sparsity.
        """
        if global_step < self.num_edge_sparsity_warmup_steps:
            fraction = global_step / self.num_edge_sparsity_warmup_steps
            if self.warmup_type == 'linear':
                return (
                    self.start_edge_sparsity
                    + (self.target_edge_sparsity - self.start_edge_sparsity) * fraction
                )
            elif self.warmup_type == 'logarithmic':
                # example: 1 - e^(log(1 - start) + fraction*(log(1 - target)-log(1 - start)))
                start, target = self.start_edge_sparsity, self.target_edge_sparsity
                log_one_minus_start = math.log(1 - start)
                log_one_minus_target = math.log(1 - target)
                val = log_one_minus_start + fraction*(log_one_minus_target - log_one_minus_start)
                return 1 - math.exp(val)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_edge_sparsity

    def get_current_node_target_sparsity(self, global_step):
        if global_step < self.num_node_sparsity_warmup_steps:
            fraction = global_step / self.num_node_sparsity_warmup_steps
            if self.warmup_type == 'linear':
                return (
                    self.start_node_sparsity
                    + (self.target_node_sparsity - self.start_node_sparsity) * fraction
                )
            elif self.warmup_type == 'logarithmic':
                start, target = self.start_node_sparsity, self.target_node_sparsity
                log_one_minus_start = math.log(1 - start)
                log_one_minus_target = math.log(1 - target)
                val = log_one_minus_start + fraction*(log_one_minus_target - log_one_minus_start)
                return 1 - math.exp(val)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_node_sparsity

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, **kwargs):
        # 1) Unpack the “reference” inputs
        idxes = inputs.pop("idxes")
        corr_input_ids = inputs.pop("corr_input_ids")
        labels = inputs.pop("labels", None)
        input_ids = inputs.pop("input_ids")


        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    labels=None,
                    **inputs
                )
                ref_logits = ref_outputs.logits


                corr_out = self.ref_model(
                    input_ids=corr_input_ids,
                    labels=None,
                    **inputs,
                    output_writer_states=True
                )
                corr_x = corr_out.writer_states
        else:
            ref_logits = None
            corr_x = None


        outputs = model(
            input_ids=input_ids,
            labels=labels,
            corr_x=corr_x,
            target_edge_sparsity=self.get_current_edge_target_sparsity(self.state.global_step),
            target_node_sparsity=None if self.disable_node_loss else self.get_current_node_target_sparsity(self.state.global_step),
            **inputs
        )

        logits = outputs["logits"]
        device = logits.device
        dtype = logits.dtype


        edge_loss = outputs.get("edge_loss", torch.tensor(0.0, device=device, dtype=dtype))
        node_loss = outputs.get("node_loss", torch.tensor(0.0, device=device, dtype=dtype))
        if self.disable_node_loss or (
                self.skip_node_loss_if_higher_sparsity and outputs["model_node_sparsity"] > outputs[
            "target_node_sparsity"]
        ):
            node_loss = torch.tensor(0.0, device=device, dtype=dtype)
        reg_loss = edge_loss + node_loss


        kl_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if ref_logits is not None:
            for i in range(logits.shape[0]):
                idx_i = idxes[i].item()
                if idx_i >= logits.shape[1]:
                    continue
                student_logp = nn.functional.log_softmax(logits[i, idx_i], dim=-1)
                teacher_logp = nn.functional.log_softmax(ref_logits[i, idx_i], dim=-1)
                kl_loss += nn.functional.kl_div(student_logp, teacher_logp, reduction='sum', log_target=True)
            kl_loss /= logits.shape[0]


        total_loss = (kl_loss + reg_loss)
        outputs["loss"] = total_loss
        outputs["kl_loss"] = kl_loss


        logger.info(
            f"Step {self.state.global_step} | loss={total_loss.item():.4f} | kl={kl_loss.item():.4f} | e_loss={edge_loss.item():.4f} | n_loss={node_loss.item():.4f} "
            f"| e_spar={outputs['model_edge_sparsity'].item():.4f} | n_spar={outputs['model_node_sparsity'].item():.4f} "
            f"| e_tgt={outputs['target_edge_sparsity'].item():.4f} | n_tgt={outputs['target_node_sparsity'].item():.4f}"
        )
        return (total_loss, outputs) if return_outputs else total_loss


@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(default="./data/datasets/some_dataset/")
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    max_seq_length: Optional[int] = field(default=72)
    start_edge_sparsity: Optional[float] = field(default=0.0)
    target_edge_sparsity: Optional[float] = field(default=1.2)
    start_node_sparsity: Optional[float] = field(default=0.0)
    target_node_sparsity: Optional[float] = field(default=0.70)
    stop_optimizing_node_if_higher_sparsity: Optional[bool] = field(default=False)
    num_sparsity_warmup_steps: Optional[int] = field(default=0)
    edge_learning_rate: Optional[float] = field(default=1e-2)
    node_learning_rate: Optional[float] = field(default=1)
    reg_edge_learning_rate: Optional[float] = field(default=1e-2)
    reg_node_learning_rate: Optional[float] = field(default=1)
    warmup_type: Optional[str] = field(default="linear")

@dataclass
class ModelArguments:
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    token: str = field(default=None)
    trust_remote_code: bool = field(default=True)
    ignore_mismatched_sizes: bool = field(default=False)
    initialize_from: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        metadata={"help": "Which QWen model to start from."},
    )
    ref_initialize_from: str = field(
        default=None,
        metadata={"help": "Optionally a second model for reference."},
    )
    with_embedding_nodes: bool = field(default=False)
    disable_linear_regularization_term: bool = field(default=False)
    disable_node_loss: bool = field(default=False)

def load_datasets(dataset_path, max_train_samples, max_eval_samples):
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path)

    if max_train_samples is not None and max_train_samples < len(dataset["train"]):
        dataset["train"] = dataset["train"].select(range(max_train_samples))
    if "validation" not in dataset:
        dataset = DatasetDict({
            "train": dataset["train"],
            "validation": dataset["train"]
        })
    if max_eval_samples is not None and max_eval_samples < len(dataset["validation"]):
        dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
    return dataset

def format_fewshot(entry):
    """
    Example "formatting" for each example.
    Modify to your real text prompt & corruption logic.
    """
    shot1 = "Q: 2+2? A: 4"
    shot2 = "Q: 3+5? A: 8"
    text_in = f"{shot1}\n{shot2}\nQ: {entry['question']} A:"
    # 'corr_input' might be some corrupted version
    corr_text = f"{shot1}\n{shot2}\nQ: {entry['question']} CORRUPT:"
    return text_in, corr_text, entry["answer"].strip()

class DataCollatorQwen:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        input_ids_list = []
        corr_input_ids_list = []
        labels_list = []
        idxes_list = []

        for ex in examples:
            prompt, corr_prompt, target_str = format_fewshot(ex)

            idx = len(self.tokenizer.encode(prompt)) - 1

            enc_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            enc_corr = self.tokenizer(
                corr_prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

            target_ids = self.tokenizer.convert_tokens_to_ids(target_str)
            if isinstance(target_ids, list):

                target_ids = target_ids[0] if len(target_ids) > 0 else 0

            input_ids_list.append(enc_prompt["input_ids"][0])
            corr_input_ids_list.append(enc_corr["input_ids"][0])
            labels_list.append(target_ids)
            idxes_list.append(idx)

        batch = {
            "input_ids": torch.stack(input_ids_list),
            "corr_input_ids": torch.stack(corr_input_ids_list),
            "labels": torch.LongTensor(labels_list),
            "idxes": torch.LongTensor(idxes_list),
        }
        return batch

def freeze_all_except_pruning_params(model):
    """Freeze all parameters except the gating (log_alphas, etc.) so that only the gating updates."""
    for n, p in model.named_parameters():
        if ('log_alpha' in n) or ('sparsity_lambda' in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

def get_optimizers(model, edges_lr, nodes_lr, reg_edges_lr, reg_nodes_lr, num_training_steps, warmup_steps=0):

    optimizer_attn_write = []
    optimizer_attn_read = []
    optimizer_edge_reg = []
    optimizer_node_reg = []

    for n, p in model.named_parameters():

        if 'read_log_alpha' in n:
            optimizer_attn_read.append(p)
        elif 'write_log_alpha' in n:
            optimizer_attn_write.append(p)
        elif 'sparsity_lambda_edges_1' in n or 'sparsity_lambda_edges_2' in n:
            optimizer_edge_reg.append(p)
        elif 'sparsity_lambda_nodes_1' in n or 'sparsity_lambda_nodes_2' in n:
            optimizer_node_reg.append(p)

    optimizer = AdamW(
        [
            {'params': optimizer_attn_write, 'lr': edges_lr},
            {'params': optimizer_attn_read, 'lr': nodes_lr},
            {'params': optimizer_edge_reg, 'maximize': True, 'lr': reg_edges_lr},
            {'params': optimizer_node_reg, 'maximize': True, 'lr': reg_nodes_lr},
        ],
        lr=edges_lr
    )
    # standard linear warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    return optimizer, scheduler


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.token is not None:

        pass

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    set_seed(training_args.seed)


    raw_datasets = load_datasets(data_args.dataset_path, data_args.max_train_samples, data_args.max_eval_samples)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]


    model = FQwenForCausalLM.from_pretrained(
        model_args.initialize_from,
        with_embedding_nodes=model_args.with_embedding_nodes,
        disable_linear_regularization_term=model_args.disable_linear_regularization_term,
        trust_remote_code=model_args.trust_remote_code
    )


    if model_args.ref_initialize_from is not None:
        ref_model = FQwenForCausalLM.from_pretrained(
            model_args.ref_initialize_from,
            with_embedding_nodes=model_args.with_embedding_nodes,
            disable_linear_regularization_term=model_args.disable_linear_regularization_term,
            trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_model = None


    model.reset_all_log_alphas()
    if ref_model is not None:
        ref_model.reset_all_log_alphas()


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.initialize_from,
        trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    freeze_all_except_pruning_params(model)


    data_collator = DataCollatorQwen(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length
    )


    trainer = FQwenTrainer(
        model=model,
        ref_model=ref_model,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        start_edge_sparsity=data_args.start_edge_sparsity,
        target_edge_sparsity=data_args.target_edge_sparsity,
        start_node_sparsity=data_args.start_node_sparsity,
        target_node_sparsity=data_args.target_node_sparsity,
        skip_node_loss_if_higher_sparsity=data_args.stop_optimizing_node_if_higher_sparsity,
        num_sparsity_warmup_steps=data_args.num_sparsity_warmup_steps,
        warmup_type=data_args.warmup_type,
        edges_lr=data_args.edge_learning_rate,
        nodes_lr=data_args.node_learning_rate,
        reg_edges_lr=data_args.reg_edge_learning_rate,
        reg_nodes_lr=data_args.reg_node_learning_rate,
        warmup_steps=training_args.warmup_steps,
        disable_node_loss=model_args.disable_node_loss,
    )


    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        logger.info(f"Eval metrics: {metrics}")

    if training_args.push_to_hub:
        trainer.push_to_hub()

def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutput

import numpy as np

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    ModelOutput
)
from dataclasses import dataclass

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

logger = logging.get_logger(__name__)


LIMIT_LEFT = -0.1
LIMIT_RIGHT = 1.1
EPS = 1e-8 # 1e-6
TEMPERATURE = 2 / 3
FACTOR = 0.8

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

class FQwenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

ALL_LAYERNORM_LAYERS.append(FQwenRMSNorm)

class FQwenRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings=131072, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @property
    def sin_cached(self):
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._cos_cached

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def cdf_stretched_concrete(x, log_alpha):
    x_01 = (x - LIMIT_LEFT) / (LIMIT_RIGHT - LIMIT_LEFT)
    intermediate = math.log(x_01) - math.log(1 - x_01)

    precursor = TEMPERATURE * intermediate - log_alpha

    prob_unclamped = torch.sigmoid(precursor)
    prob_clamped = torch.clamp(prob_unclamped, EPS, 1 - EPS)
    return prob_clamped


def sample_z_from_u(u, log_alpha):
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / TEMPERATURE)
    return (LIMIT_RIGHT - LIMIT_LEFT) * s + LIMIT_LEFT

def sample_z_from_log_alpha(log_alpha):
    u = torch.autograd.Variable(torch.empty(log_alpha.shape, dtype=log_alpha.dtype).uniform_(EPS, 1 - EPS)).to(
        log_alpha.device)
    z = sample_z_from_u(u, log_alpha)
    z = F.hardtanh(z, 0, 1)

    return z


def deterministic_z_from_log_alpha(log_alpha, apply_one=False):
    size = np.prod(log_alpha.shape)


    csc = cdf_stretched_concrete(0, log_alpha)
    expected_num_nonzeros = torch.sum(1 - csc)
    expected_num_zeros = size - expected_num_nonzeros
    num_zeros = int(torch.round(expected_num_zeros).item())

    soft_mask = torch.sigmoid(log_alpha / TEMPERATURE * FACTOR).reshape(-1)

    if num_zeros > 0:
        if soft_mask.ndim == 0:
            soft_mask = torch.tensor(0).to(log_alpha.device)
        else:
            _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
            soft_mask[indices] = 0
            if apply_one:
                soft_mask[soft_mask > 0] = 1
    return soft_mask.reshape(log_alpha.shape)



def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def num_writers(config, with_embedding_nodes=False):
    n_writers = 1 if with_embedding_nodes else 0
    for l in range(config.num_hidden_layers):
        n_writers += config.num_attention_heads + 1

    return n_writers

def num_readers(config):

    n_readers = 0
    for l in range(config.num_hidden_layers):
        n_readers += (config.num_attention_heads + 2 * config.num_key_value_heads) + 1
    n_readers += 1
    return n_readers

def num_edges(config, with_embedding_nodes=False):
    n_edges = num_readers(config) if with_embedding_nodes else 0
    for l in range(config.num_hidden_layers):
        n_edges += config.num_attention_heads * (
            1 +
            (config.num_hidden_layers - l - 1) * (config.num_attention_heads + 2 * config.num_key_value_heads + 1) +
            1
        )
        n_edges += (config.num_hidden_layers - l - 1) * (config.num_attention_heads + 2 * config.num_key_value_heads + 1) + 1

    return n_edges

def num_nodes(config, with_embedding_nodes=False):
    return num_writers(config, with_embedding_nodes)


def writer_idx_to_name(writer_idx, num_layers, num_heads, with_embedding_nodes=False):
    if with_embedding_nodes:
        if writer_idx == 0:
            return "embeds"
        else:
            writer_idx -= 1

    layer_idx = writer_idx // (num_heads + 1)
    head_idx = writer_idx % (num_heads + 1)
    if head_idx == num_heads:
        return f"m{layer_idx}"
    else:
        return f"a{layer_idx}.h{head_idx}"

def writer_name_to_idx(name, num_layers, num_heads, with_embedding_nodes=False):
    idx = 0
    if with_embedding_nodes:
        if name == "embeds":
            return 0
        else:
            idx += 1
    if name.startswith("m"):
        layer_idx = int(name[1:])
        idx += layer_idx * (num_heads + 1) + num_heads
    elif name.startswith("a"):
        parts = name.split(".")
        layer_idx = int(parts[0][1:])
        head_idx = int(parts[1][1:])
        idx += layer_idx * (num_heads + 1) + head_idx
    else:
        raise ValueError(f"Unrecognized writer name {name}")
    return idx

def reader_idx_to_name(reader_idx, num_layers, num_heads, num_key_value_heads):
    layer_idx = reader_idx // (num_heads + 2 * num_key_value_heads + 1)
    head_idx = reader_idx % (num_heads + 2 * num_key_value_heads + 1)
    if layer_idx == num_layers:
        return "resid_post"

    if head_idx < num_heads:
        return f"a{layer_idx}.h{head_idx}.q"
    elif head_idx < num_heads + num_key_value_heads:
        return f"a{layer_idx}.h{head_idx - num_heads}.k"
    elif head_idx < num_heads + 2 * num_key_value_heads:
        return f"a{layer_idx}.h{head_idx - num_heads - num_key_value_heads}.v"
    else:
        return f"m{layer_idx}"

def get_mask(log_alpha, training=False, threshold_for_deterministic=None, apply_one=False):
    if training:
        mask = sample_z_from_log_alpha(log_alpha)
    else:
        mask = deterministic_z_from_log_alpha(log_alpha, apply_one=apply_one)
        if threshold_for_deterministic is not None:
            mask = (mask > threshold_for_deterministic).to(mask.dtype)
    return mask


class FQwenMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if not hasattr(self.config, "pretraining_tp"):
            self.config.pretraining_tp = 1
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class FQwen2Attention(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = FQwenRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = FQwenLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = FQwenDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


    def _apply_headwise_linear(self, x, weight, num_heads):
        _, bsz, seq_len, _ = x.shape
        weight_ = weight.view(num_heads, self.head_dim, self.hidden_size)
        projected = torch.einsum(
            'nbld,nhd->nblh',
            x,
            weight_
        )
        projected = projected.permute(1, 0, 2, 3)
        return projected


    def _apply_output_linear(self, x, weight, num_heads):
        bsz, _, seq_len, _ = x.shape
        weight_ = weight.view(self.hidden_size, num_heads, self.head_dim)
        projected = torch.einsum(
            'bnlh,dnh->nbld',
            x,
            weight_
        )

        return projected

    def forward(
        self,
        q_hidden_states: torch.Tensor,
        k_hidden_states: torch.Tensor,
        v_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        _, bsz, q_len, _ = q_hidden_states.size()

        if self.config.pretraining_tp > 1:
            assert self.num_heads % self.config.pretraining_tp == 0, "Number of heads must be divisible by pretraining_tp"
            assert self.num_key_value_heads % self.config.pretraining_tp == 0, "Number of key value heads must be divisible by pretraining_tp"
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            q_hidden_states = q_hidden_states.split(self.num_heads // self.config.pretraining_tp, dim=-1)
            query_states = [
                self._apply_headwise_linear(
                    q_hidden_states[i],
                    query_slices[i],
                    self.num_heads // self.config.pretraining_tp
                )
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=1)

            k_hidden_states = k_hidden_states.split(self.num_key_value_heads // self.config.pretraining_tp, dim=-1)
            key_states = [
                self._apply_headwise_linear(
                    k_hidden_states[i],
                    key_slices[i],
                    self.num_heads // self.config.pretraining_tp
                )
                for i in range(self.config.pretraining_tp)
            ]

            key_states = torch.cat(key_states, dim=1)

            v_hidden_states = v_hidden_states.split(self.num_key_value_heads // self.config.pretraining_tp, dim=-1)
            value_states = [
                self._apply_headwise_linear(
                    v_hidden_states[i],
                    value_slices[i],
                    self.num_heads // self.config.pretraining_tp
                )
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=1)

        else:
            query_states = self._apply_headwise_linear(q_hidden_states, self.q_proj.weight, self.num_heads)
            key_states = self._apply_headwise_linear(k_hidden_states, self.k_proj.weight, self.num_key_value_heads)
            value_states = self._apply_headwise_linear(v_hidden_states, self.v_proj.weight, self.num_key_value_heads)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.num_heads // self.config.pretraining_tp, dim=1)
            o_proj_slices = self.o_proj.weight.split(self.num_heads * self.head_dim // self.config.pretraining_tp,
                                                     dim=1)
            attn_output = sum([
                self._apply_output_linear(
                    attn_output[i],
                    o_proj_slices[i],
                    self.num_heads // self.config.pretraining_tp
                ) for i in range(self.config.pretraining_tp)
            ])
        else:
            attn_output = self._apply_output_linear(attn_output, self.o_proj.weight, self.num_heads)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class FQwenFlashAttention2(FQwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def _apply_output_linear_flash(self, x, weight, num_heads):
        bsz, seq_len, _, _ = x.shape
        weight_ = weight.view(self.hidden_size, num_heads, self.head_dim)
        projected = torch.einsum(
            'blnh,dnh->nbld',
            x,
            weight_
        )
        return projected

    def forward(
        self,
        q_hidden_states: torch.Tensor,
        k_hidden_states: torch.Tensor,
        v_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        _, bsz, q_len, _ = q_hidden_states.size()

        query_states = self._apply_headwise_linear(q_hidden_states, self.q_proj.weight, self.num_heads)
        key_states = self._apply_headwise_linear(k_hidden_states, self.k_proj.weight, self.num_key_value_heads)
        value_states = self._apply_headwise_linear(v_hidden_states, self.v_proj.weight, self.num_key_value_heads)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = self._apply_output_linear_flash(attn_output, self.o_proj.weight, self.num_heads)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class FQwenSdpaAttention(FQwen2Attention):
    def forward(
        self,
        q_hidden_states: torch.Tensor,
        k_hidden_states: torch.Tensor,
        v_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                q_hidden_states=q_hidden_states,
                k_hidden_states=k_hidden_states,
                v_hidden_states=v_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        _, bsz, q_len, _ = q_hidden_states.size()

        query_states = self._apply_headwise_linear(q_hidden_states, self.q_proj.weight, self.num_heads)
        key_states = self._apply_headwise_linear(k_hidden_states, self.k_proj.weight, self.num_key_value_heads)
        value_states = self._apply_headwise_linear(v_hidden_states, self.v_proj.weight, self.num_key_value_heads)

        cos, sin = self.rotary_emb(value_states, position_ids)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = self._apply_output_linear(attn_output, self.o_proj.weight, self.num_heads)

        return attn_output, None, past_key_value


FQwen_ATTENTION_CLASSES = {
    "eager": FQwen2Attention,
    "flash_attention_2": FQwenFlashAttention2,
    "sdpa": FQwenSdpaAttention,
}


class FQwenDecoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig,
                 layer_idx: int,
                 with_embedding_nodes=False):

        super().__init__()
        self.hidden_size = config.hidden_size

        if not hasattr(config, "attention_bias"):
            config.attention_bias = False
        self.self_attn = FQwen_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = FQwenMLP(config)
        self.input_layernorm = FQwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FQwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_writers = num_writers(config, with_embedding_nodes=with_embedding_nodes)
        self.num_readers = num_readers(config)
        self.edge_threshold_for_deterministic = None
        self.node_threshold_for_deterministic = None
        self._dtype = self.mlp.gate_proj.weight.dtype

        writer_offset = 1 if with_embedding_nodes else 0
        self.attn_writer_idx = writer_offset + layer_idx * (self.num_heads + 1)
        self.attn_reader_idx = layer_idx * (self.num_heads + 2 * self.num_kv_heads + 1)
        self.mlp_writer_idx = writer_offset + (layer_idx + 1) * (self.num_heads + 1) - 1

        self.mlp_reader_idx = (layer_idx + 1) * (self.num_heads + 2 * self.num_kv_heads + 1) - 1

        self.q_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_heads, dtype=self._dtype))
        self.k_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_kv_heads, dtype=self._dtype))
        self.v_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_kv_heads, dtype=self._dtype))
        self.attn_write_log_alphas = nn.Parameter(torch.empty(self.num_heads, dtype=self._dtype))
        self.q_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.k_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.v_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.attn_write_log_alphas.data.normal_(mean=10.0, std=0.01)

        attn_read_common_mask = torch.zeros(self.num_writers, dtype=self._dtype)
        attn_read_common_mask[:self.attn_writer_idx] = 1
        attn_read_common_mask = attn_read_common_mask.unsqueeze(1)
        self.register_buffer("attn_read_common_mask", attn_read_common_mask)

        attn_write_common_mask = F.pad(
            torch.eye(self.num_heads, dtype=torch.float32).to(self._dtype),
            (self.attn_writer_idx, self.num_writers - self.attn_writer_idx - self.num_heads, 0, 0)
        )
        self.register_buffer("attn_write_common_mask", attn_write_common_mask)

        self.mlp_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, dtype=self._dtype))
        self.mlp_write_log_alphas = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        self.mlp_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_write_log_alphas.data.normal_(mean=10.0, std=0.01)

        mlp_read_common_mask = torch.zeros(self.num_writers, dtype=self._dtype)
        mlp_read_common_mask[:self.mlp_writer_idx] = 1
        self.register_buffer("mlp_read_common_mask", mlp_read_common_mask)

        mlp_write_common_mask = torch.zeros((self.num_writers, 1), dtype=self._dtype)
        mlp_write_common_mask[self.mlp_writer_idx, 0] = 1
        self.register_buffer("mlp_write_common_mask", mlp_write_common_mask)

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.edge_threshold_for_deterministic = edge_threshold_for_deterministic

    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.node_threshold_for_deterministic = node_threshold_for_deterministic

    @torch.no_grad()
    def get_edge_masks(self):
        z_q = get_mask(
            self.q_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic
        )
        z_q = z_q[:self.attn_writer_idx, :]
        z_k = get_mask(
            self.k_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic
        )
        z_k = z_k[:self.attn_writer_idx, :]
        z_v = get_mask(
            self.v_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic
        )
        z_v = z_v[:self.attn_writer_idx, :]

        z_mlp = get_mask(
            self.mlp_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic
        )
        z_mlp = z_mlp[:self.mlp_writer_idx]

        return (z_q, z_k, z_v, z_mlp)

    @torch.no_grad()
    def get_node_masks(self):
        z_attn = get_mask(
            self.attn_write_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.node_threshold_for_deterministic
        )

        z_mlp = get_mask(
            self.mlp_write_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.node_threshold_for_deterministic
        ).reshape([])

        return (z_attn, z_mlp)

    @torch.no_grad()
    def reset_all_log_alphas(self):
        self.q_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.k_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.v_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.attn_write_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_write_log_alphas.data.normal_(mean=10.0, std=0.01)

    @torch.no_grad()
    def load_attn_log_alphas(self, attn_in_edges):
        self.q_read_log_alphas.data.fill_(-10)
        self.k_read_log_alphas.data.fill_(-10)
        self.v_read_log_alphas.data.fill_(-10)

        for writer_idx, reader in attn_in_edges:
            reader_portions = reader.split(".")
            assert len(reader_portions) == 3, f"Invalid reader format: {reader}"
            layer_idx = int(reader_portions[0][1:])
            head = int(reader_portions[1][1:])
            qkv = reader_portions[2]
            assert layer_idx == self.layer_idx, f"Invalid layer index: {layer_idx}"
            if qkv == "q":
                self.q_read_log_alphas[writer_idx, head] = 10
            elif qkv == "k":
                self.k_read_log_alphas[writer_idx, head] = 10
            elif qkv == "v":
                self.v_read_log_alphas[writer_idx, head] = 10

        self.attn_write_log_alphas.data.fill_(10)

    @torch.no_grad()
    def load_mlp_log_alphas(self, mlp_in_edges):
        self.mlp_read_log_alphas.data.fill_(-10)

        for writer_idx, reader in mlp_in_edges:
            reader_portions = reader.split(".")
            assert len(reader_portions) == 1, f"Invalid reader format: {reader}"
            layer_idx = int(reader_portions[0][1:])
            assert layer_idx == self.layer_idx, f"Invalid layer index: {layer_idx}"
            self.mlp_read_log_alphas[writer_idx] = 10


        self.mlp_write_log_alphas.data.fill_(10)

    def attn_read(
        self,
        hidden_states: torch.Tensor,
        corr_x: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None
    ):

        q_m = get_mask(
            self.q_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic
        )
        k_m = get_mask(
            self.k_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic
        )
        v_m = get_mask(
            self.v_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic
        )

        q_z = q_m * self.attn_read_common_mask  # (w, h)
        k_z = k_m * self.attn_read_common_mask
        v_z = v_m * self.attn_read_common_mask

        x_q = torch.einsum("wbsd,wh->hbsd", hidden_states, q_z)
        x_k = torch.einsum("wbsd,wh->hbsd", hidden_states, k_z)
        x_v = torch.einsum("wbsd,wh->hbsd", hidden_states, v_z)

        if embeds is not None:
            x_q = x_q + embeds.unsqueeze(0)
            x_k = x_k + embeds.unsqueeze(0)
            x_v = x_v + embeds.unsqueeze(0)

        if corr_x is not None:
            x_q = x_q + torch.einsum("wbsd,wh->hbsd", corr_x, (1-q_m) * self.attn_read_common_mask)
            x_k = x_k + torch.einsum("wbsd,wh->hbsd", corr_x, (1-k_m) * self.attn_read_common_mask)
            x_v = x_v + torch.einsum("wbsd,wh->hbsd", corr_x, (1-v_m) * self.attn_read_common_mask)

        z_edges_sum = torch.sum(q_z) + torch.sum(k_z) + torch.sum(v_z)

        return x_q, x_k, x_v, z_edges_sum


    def attn_write(
        self,
        residual: torch.Tensor,
        attn_out: torch.Tensor,
        corr_x: Optional[torch.Tensor] = None
    ):
        z = get_mask(
            self.attn_write_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.node_threshold_for_deterministic
        ).reshape(-1, 1, 1, 1)

        x = attn_out * z

        if corr_x is not None:
            x = x + corr_x[self.attn_writer_idx : self.attn_writer_idx + self.num_heads] * (1-z)

        x = torch.einsum("nbsd,nw->wbsd", x, self.attn_write_common_mask)

        residual = residual + x
        z_nodes_sum = torch.sum(z)

        return residual, z_nodes_sum


    def mlp_read(
        self,
        hidden_states: torch.Tensor,
        corr_x: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None
    ):
        m = get_mask(self.mlp_read_log_alphas, training=self.training,
                     threshold_for_deterministic=self.edge_threshold_for_deterministic)

        z = m * self.mlp_read_common_mask

        x_z = torch.einsum("wbsd,w->bsd", hidden_states, z)

        if embeds is not None:
            x_z = x_z + embeds
        if corr_x is not None:
            x_z = x_z + torch.einsum("wbsd,w->bsd", corr_x, (1-m) * self.mlp_read_common_mask)

        z_edges_sum = torch.sum(z)

        return x_z, z_edges_sum


    def mlp_write(
        self,
        residual: torch.Tensor,
        mlp_out: torch.Tensor,
        corr_x: Optional[torch.Tensor] = None
    ):
        z = get_mask(
            self.mlp_write_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.node_threshold_for_deterministic
        ).reshape(1, 1, 1)

        x = mlp_out * z

        if corr_x is not None:
            x = x + corr_x[self.mlp_writer_idx] * (1-z)

        x = torch.einsum("ibsd,wi->wbsd", x.unsqueeze(0), self.mlp_write_common_mask)
        residual = residual + x

        return residual, torch.sum(z)


    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            corr_x: Optional[torch.Tensor] = None,
            embeds: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        q_hidden_states, k_hidden_states, v_hidden_states, z_attn_edges_sum = self.attn_read(hidden_states,
                                                                                             corr_x=corr_x,
                                                                                             embeds=embeds)
        q_hidden_states = self.input_layernorm(q_hidden_states)
        k_hidden_states = self.input_layernorm(k_hidden_states)
        v_hidden_states = self.input_layernorm(v_hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            q_hidden_states=q_hidden_states,
            k_hidden_states=k_hidden_states,
            v_hidden_states=v_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        residual, z_attn_nodes_sum = self.attn_write(residual, hidden_states, corr_x=corr_x)
        hidden_states, z_mlp_edges_sum = self.mlp_read(residual, corr_x=corr_x, embeds=embeds)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states, z_mlp_nodes_sum = self.mlp_write(residual, hidden_states, corr_x=corr_x)


        z_edges_sum = z_attn_edges_sum + z_mlp_edges_sum
        z_nodes_sum = z_attn_nodes_sum + z_mlp_nodes_sum

        outputs = (hidden_states, z_edges_sum, z_nodes_sum)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs



class FQwenPreTrainedModel(PreTrainedModel):
    config_class = PretrainedConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FQwenDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            if hasattr(self.config, "_pre_quantization_dtype"):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = layer.self_attn.o_proj.weight.dtype
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, device=device, dtype=dtype
            )

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


@dataclass
class FQwenModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    writer_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_edge_sparsity: Optional[torch.FloatTensor] = None
    target_node_sparsity: Optional[torch.FloatTensor] = None
    model_edge_sparsity: Optional[torch.FloatTensor] = None
    model_node_sparsity: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None


class FQwenModel(FQwenPreTrainedModel):
    def __init__(self,
                 config: PretrainedConfig,
                 with_embedding_nodes=False,
                 disable_linear_regularization_term=False,):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                FQwenDecoderLayer(config, layer_idx, with_embedding_nodes=with_embedding_nodes)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = FQwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_writers = num_writers(config, with_embedding_nodes=with_embedding_nodes)
        self.num_readers = num_readers(config)
        self.num_layers = config.num_hidden_layers
        self.num_edges = num_edges(config, with_embedding_nodes=with_embedding_nodes)
        self.num_nodes = num_nodes(config, with_embedding_nodes=with_embedding_nodes)
        self.edge_threshold_for_deterministic = None
        self.node_threshold_for_deterministic = None
        self._dtype = self.norm.weight.dtype
        self.with_embedding_nodes = with_embedding_nodes

        if self.with_embedding_nodes:
            self.token_write_log_alpha = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
            self.token_write_log_alpha.data.normal_(mean=10.0, std=0.01)

            token_write_mask = torch.zeros(self.num_writers, dtype=self._dtype)
            token_write_mask[0] = 1
            self.register_buffer("token_write_mask", token_write_mask)

        self.final_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, dtype=self._dtype))
        self.final_read_log_alphas.data.normal_(mean=10.0, std=0.01)

        if disable_linear_regularization_term:
            self.sparsity_lambda_edges_1 = torch.tensor([0.0], dtype=self._dtype)
            self.sparsity_lambda_nodes_1 = torch.tensor([0.0], dtype=self._dtype)
        else:
            self.sparsity_lambda_edges_1 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
            self.sparsity_lambda_nodes_1 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        self.sparsity_lambda_edges_2 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        self.sparsity_lambda_nodes_2 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))

        self.post_init()


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.edge_threshold_for_deterministic = edge_threshold_for_deterministic

        for layer in self.layers:
            layer.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)


    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.node_threshold_for_deterministic = node_threshold_for_deterministic

        for layer in self.layers:
            layer.set_node_threshold_for_deterministic(node_threshold_for_deterministic)


    @torch.no_grad()
    def get_edge_masks(self):
        masks = []
        for layer in self.layers:
            masks.append(layer.get_edge_masks())
        z_final = get_mask(self.final_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        masks.append((z_final,))
        return masks

    @torch.no_grad()
    def get_node_masks(self):
        masks = []
        if self.with_embedding_nodes:
            z_tokens = get_mask(
                self.token_write_log_alpha,
                training=self.training,
                threshold_for_deterministic=self.node_threshold_for_deterministic
            ).reshape([])
            masks.append((z_tokens,))
        for layer in self.layers:
            masks.append(layer.get_node_masks())
        return masks

    @torch.no_grad()
    def get_edge_sparsity(self):
        edge_masks = self.get_edge_masks()
        def process(mask):
            return torch.sum(mask), torch.numel(mask)
        s, n = 0, 0
        for l in range(self.num_layers):
            for i in range(4):
                s_, n_ = process(edge_masks[l][i])
                s += s_
                n += n_

        s_, n_ = process(edge_masks[-1][0])
        s += s_
        n += n_

        s /= (1 if n == 0 else n)
        return 1 - s

    @torch.no_grad()
    def get_node_sparsity(self):
        node_masks = self.get_node_masks()
        def process(mask):
            return torch.sum(mask), torch.numel(mask)
        s, n = 0, 0
        if self.with_embedding_nodes:
            s_, n_ = process(node_masks[0][0])
            s += s_
            n += n_
            offset = 1
        else:
            offset = 0
        for l in range(len(self.layers)):
            for i in range(2):
                s_, n_ = process(node_masks[l+offset][i])
                s += s_
                n += n_

        s /= (1 if n == 0 else n)
        return 1 - s

    @torch.no_grad()
    def get_effective_edge_sparsity(self):
        edge_masks = self.get_edge_masks()
        node_masks = self.get_node_masks()

        full_node_mask = torch.cat([mask.reshape(-1) for group in node_masks for mask in group], dim=0)

        def process(mask):
            mask = mask * full_node_mask[:mask.shape[0]].reshape(-1, *([1] * (mask.ndim - 1)))
            return torch.sum(mask), torch.numel(mask)

        s, n = 0, 0
        for l in range(self.num_layers):
            for i in range(4):
                s_, n_ = process(edge_masks[l][i])
                s += s_
                n += n_

        s_, n_ = process(edge_masks[-1][0])
        s += s_
        n += n_

        s /= (1 if n == 0 else n)
        return 1 - s

    @torch.no_grad()
    def get_edges(self):
        edge_masks = self.get_edge_masks()
        node_masks = self.get_node_masks()

        allowed_writers = []
        edges = []

        if self.with_embedding_nodes:
            if node_masks[0][0] == 1:
                allowed_writers.append(0)
            offset = 1
            layer_offset = 1
        else:
            offset = 0
            layer_offset = 0

        for l in range(self.num_layers):
            attn_writers = node_masks[l + layer_offset][0]
            for i in range(self.num_heads):
                if attn_writers[i] == 1:
                    allowed_writers.append(offset + l * (1 + self.num_heads) + i)
            mlp_writers = node_masks[l + layer_offset][1]
            if mlp_writers == 1:
                allowed_writers.append(offset + (l + 1) * (1 + self.num_heads) - 1)

            attn_q_edges, attn_k_edges, attn_v_edges, mlp_edges = edge_masks[l]
            for from_idx in range(attn_q_edges.shape[0]):
                if from_idx not in allowed_writers:
                    continue
                for head_no in range(attn_q_edges.shape[1]):
                    if attn_q_edges[from_idx, head_no] == 1:
                        to_idx = l * (1 + self.num_heads + 2 * self.num_kv_heads) + head_no
                        edges.append((
                            writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads,
                                               with_embedding_nodes=self.with_embedding_nodes),
                            reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads,
                                               num_key_value_heads=self.num_kv_heads)
                        ))
                for head_no in range(attn_k_edges.shape[1]):
                    if attn_k_edges[from_idx, head_no] == 1:
                        to_idx = l * (1 + self.num_heads + 2 * self.num_kv_heads) + self.num_heads + head_no
                        edges.append((
                            writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads,
                                               with_embedding_nodes=self.with_embedding_nodes),
                            reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads,
                                               num_key_value_heads=self.num_kv_heads)
                        ))
                for head_no in range(attn_v_edges.shape[1]):
                    if attn_v_edges[from_idx, head_no] == 1:
                        to_idx = l * (
                                    1 + self.num_heads + 2 * self.num_kv_heads) + self.num_heads + self.num_kv_heads + head_no
                        edges.append((
                            writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads,
                                               with_embedding_nodes=self.with_embedding_nodes),
                            reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads,
                                               num_key_value_heads=self.num_kv_heads)
                        ))
            for from_idx in range(mlp_edges.shape[0]):
                if from_idx not in allowed_writers:
                    continue
                if mlp_edges[from_idx] == 1:
                    to_idx = (l + 1) * (1 + self.num_heads + 2 * self.num_kv_heads) - 1
                    edges.append((
                        writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads,
                                           with_embedding_nodes=self.with_embedding_nodes),
                        reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads,
                                           num_key_value_heads=self.num_kv_heads)
                    ))
        final_read_mask = edge_masks[self.num_layers][0]
        for from_idx in range(self.num_writers):
            if (from_idx in allowed_writers) and (final_read_mask[from_idx] == 1):
                edges.append((
                    writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads,
                                       with_embedding_nodes=self.with_embedding_nodes),
                    f"resid_post"
                ))
        return edges

    @torch.no_grad()
    def reset_all_log_alphas(self):
        if self.with_embedding_nodes:
            self.token_write_log_alpha.data.normal_(mean=10.0, std=0.01)
        for layer in self.layers:
            layer.reset_all_log_alphas()
        self.final_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.sparsity_lambda_edges_1.data.zero_()
        self.sparsity_lambda_nodes_1.data.zero_()


    @torch.no_grad()
    def load_resid_post_log_alphas(self, edges):
        # Fill with -10 by default
        self.final_read_log_alphas.data.fill_(-10)

        for writer_idx, reader in edges:
            assert reader == "resid_post", f"Invalid reader format: {reader}"
            self.final_read_log_alphas[writer_idx] = 10


        if self.with_embedding_nodes:
            self.token_write_log_alpha.data.fill_(10)


    @torch.no_grad()
    def load_all_log_alphas(self, edges):
        layer_attn_in_edges = [[] for _ in range(self.num_layers)]
        layer_mlp_in_edges = [[] for _ in range(self.num_layers)]
        resid_post_edges = []
        for edge in edges:
            writer, reader = edge
            writer_idx = writer_name_to_idx(writer, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes)
            if reader == "resid_post":
                resid_post_edges.append((writer_idx, reader))
            elif reader.startswith("m"):
                layer_idx = int(reader[1:])
                layer_mlp_in_edges[layer_idx].append((writer_idx, reader))
            elif reader.startswith("a"):
                layer_idx = int(reader[1:reader.find(".")])
                layer_attn_in_edges[layer_idx].append((writer_idx, reader))
            else:
                raise ValueError(f"Invalid reader format: {reader}")
        for layer_idx, attn_in_edges in enumerate(layer_attn_in_edges):
            self.layers[layer_idx].load_attn_log_alphas(attn_in_edges)
        for layer_idx, mlp_in_edges in enumerate(layer_mlp_in_edges):
            self.layers[layer_idx].load_mlp_log_alphas(mlp_in_edges)
        self.load_resid_post_log_alphas(resid_post_edges)


    def read(self, x, corr_x=None, embeds=None):

        z = get_mask(self.final_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        x_z = torch.einsum("wbsd,w->bsd", x, z)

        if embeds is not None:
            x_z = x_z + embeds
        if corr_x is not None:
            x_z = x_z + torch.einsum("wbsd,w->bsd", corr_x, (1-z))

        z_edges_sum = torch.sum(z)

        return x_z, z_edges_sum

    def write(self, tok_embeds, corr_x=None):

        if self.with_embedding_nodes:
            z_tokens = get_mask(
                self.token_write_log_alpha,
                training=self.training,
                threshold_for_deterministic=self.node_threshold_for_deterministic
            ).reshape(1, 1, 1)
            tok_embeds = tok_embeds * z_tokens
            if corr_x is not None:
                tok_embeds = tok_embeds + corr_x[0] * (1 - z_tokens)


            hidden_states = tok_embeds.detach().unsqueeze(0) * self.token_write_mask.reshape(-1, 1, 1, 1)
            z_nodes_sum = torch.sum(z_tokens)

            return hidden_states, None, z_nodes_sum
        else:
            hidden_states = torch.zeros(self.num_writers, *tok_embeds.shape, dtype=tok_embeds.dtype, device=tok_embeds.device)
            z_nodes_sum = 0
            return hidden_states, tok_embeds, z_nodes_sum


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_edge_sparsity: Optional[float] = None,
        target_node_sparsity: Optional[float] = None,
        corr_x = None,
        output_writer_states: Optional[bool] = False,
    ) -> Union[Tuple, FQwenModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_seen_tokens + inputs_embeds.shape[1]
        )

        # embed positions
        hidden_states, embeds, z_nodes_sum = self.write(inputs_embeds, corr_x=corr_x)
        z_edges_sum = 0

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    corr_x,
                    embeds,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    corr_x=corr_x,
                    embeds=embeds,
                )

            hidden_states, z_layer_edges_sum, z_layer_nodes_sum = layer_outputs[0], layer_outputs[1], layer_outputs[2]
            z_edges_sum = z_edges_sum + z_layer_edges_sum
            z_nodes_sum = z_nodes_sum + z_layer_nodes_sum

            if use_cache:
                next_decoder_cache = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_self_attns += (layer_outputs[3],)

        if output_writer_states:
            writer_states = hidden_states
        else:
            writer_states = None

        hidden_states, z_final_edges_sum = self.read(hidden_states, corr_x=corr_x, embeds=embeds)

        z_edges_sum = z_edges_sum + z_final_edges_sum
        hidden_states = self.norm(hidden_states)


        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        model_edge_sparsity = 1 - (z_edges_sum / self.num_edges)
        model_node_sparsity = 1 - (z_nodes_sum / self.num_nodes)

        if target_edge_sparsity is None:
            edge_loss = None
        else:
            edge_loss = self.sparsity_lambda_edges_1.reshape([]) * (
                    model_edge_sparsity - target_edge_sparsity
            ) + self.sparsity_lambda_edges_2.reshape([]) * (
                                model_edge_sparsity - target_edge_sparsity
                        ) ** 2

        if target_node_sparsity is None:
            node_loss = None
        else:
            node_loss = self.sparsity_lambda_nodes_1.reshape([]) * (
                    model_node_sparsity - target_node_sparsity
            ) + self.sparsity_lambda_nodes_2.reshape([]) * (
                                model_node_sparsity - target_node_sparsity
                        ) ** 2

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )

        if target_edge_sparsity is not None:
            target_edge_sparsity = torch.tensor(target_edge_sparsity, device=model_edge_sparsity.device,
                                                dtype=model_edge_sparsity.dtype)
        if target_node_sparsity is not None:
            target_node_sparsity = torch.tensor(target_node_sparsity, device=model_node_sparsity.device,
                                                dtype=model_node_sparsity.dtype)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    writer_states,
                    target_edge_sparsity,
                    target_node_sparsity,
                    model_edge_sparsity,
                    model_node_sparsity,
                    edge_loss,
                    node_loss,
                ] if v is not None
            )

        return FQwenModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            writer_states=writer_states,
            target_edge_sparsity=target_edge_sparsity,
            target_node_sparsity=target_node_sparsity,
            model_edge_sparsity=model_edge_sparsity,
            model_node_sparsity=model_node_sparsity,
            edge_loss=edge_loss,
            node_loss=node_loss,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, current_length):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else current_length + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):

            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


@dataclass
class FQwenForCausalLMOutput(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    writer_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_edge_sparsity: Optional[torch.FloatTensor] = None
    target_node_sparsity: Optional[torch.FloatTensor] = None
    model_edge_sparsity: Optional[torch.FloatTensor] = None
    model_node_sparsity: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None


class FQwenForCausalLM(FQwenPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: PretrainedConfig,
        with_embedding_nodes: bool = False,
        disable_linear_regularization_term=False,
    ):
        super().__init__(config)
        self.model = FQwenModel(
            config,
            with_embedding_nodes=with_embedding_nodes,
            disable_linear_regularization_term=disable_linear_regularization_term
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.model.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)

    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.model.set_node_threshold_for_deterministic(node_threshold_for_deterministic)

    @torch.no_grad()
    def get_edge_masks(self):
        return self.model.get_edge_masks()

    @torch.no_grad()
    def get_node_masks(self):
        return self.model.get_node_masks()

    @torch.no_grad()
    def get_edge_sparsity(self):
        return self.model.get_edge_sparsity()

    @torch.no_grad()
    def get_node_sparsity(self):
        return self.model.get_node_sparsity()

    @torch.no_grad()
    def get_effective_edge_sparsity(self):
        return self.model.get_effective_edge_sparsity()

    @torch.no_grad()
    def get_edges(self):
        return self.model.get_edges()

    @torch.no_grad()
    def reset_all_log_alphas(self):
        self.model.reset_all_log_alphas()

    @torch.no_grad()
    def load_all_log_alphas(self, edges):
        self.model.load_all_log_alphas(edges)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_edge_sparsity: Optional[float] = None,
        target_node_sparsity: Optional[float] = None,
        corr_x = None,
        output_writer_states: Optional[bool] = False,
    ) -> Union[Tuple, FQwenForCausalLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            target_edge_sparsity=target_edge_sparsity,
            target_node_sparsity=target_node_sparsity,
            corr_x=corr_x,
            output_writer_states=output_writer_states,
        )

        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # pass hidden_states into lm_head
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return FQwenForCausalLMOutput(
            lm_loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            writer_states=outputs.writer_states,
            target_edge_sparsity=outputs.target_edge_sparsity,
            target_node_sparsity=outputs.target_node_sparsity,
            model_edge_sparsity=outputs.model_edge_sparsity,
            model_node_sparsity=outputs.model_node_sparsity,
            edge_loss=outputs.edge_loss,
            node_loss=outputs.node_loss,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None,
            **kwargs
    ):

        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None


            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]

            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]


        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:

            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]


        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:

            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )


        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

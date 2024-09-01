"""PyTorch Dbrx model."""

import math
import warnings
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (MoeCausalLMOutputWithPast,
                                           MoeModelOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_flash_attn_2_available, logging

from .configuration_dbrx import DbrxAttentionConfig, DbrxConfig, DbrxFFNConfig

if is_flash_attn_2_available():
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input  # noqa
        from flash_attn.bert_padding import index_first_axis, unpad_input
    except:
        pass

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = 'DbrxConfig'

#############################################################################
# Copied from LLaMaRotaryEmbedding
#############################################################################


class DbrxRotaryEmbedding(nn.Module):

    def __init__(self,
                 dim: int,
                 max_position_embeddings: int = 2048,
                 base: float = 10000.0,
                 scaling_factor: float = 1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base**(
            torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(
            self, x: torch.Tensor, position_ids: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(
            device_type, str) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float()
                     @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos and
            sin so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos and sin have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos and sin broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch, num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


#############################################################################

#############################################################################
# Modified from modeling_mixtral
#############################################################################


def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    r"""Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts (`int`):
            Number of experts.
        top_k (`int`):
            The number of experts each token is routed to.
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return torch.tensor(0.0)

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits],
            dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits,
                                                  dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (attention_mask[None, :, :, None, None].expand(
            (num_hidden_layers, batch_size, sequence_length, top_k,
             num_experts)).reshape(-1, top_k, num_experts).to(compute_device))

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
                expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None].expand(
                (num_hidden_layers, batch_size, sequence_length,
                 num_experts)).reshape(-1, num_experts).to(compute_device))

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask,
            dim=0) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert *
                             router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


#############################################################################


def resolve_ffn_act_fn(
        ffn_act_fn: dict) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve the activation function for the feed-forward network.

    Args:
        ffn_act_fn (dict): The configuration dictionary for the activation function.
            The dict config must specify the 'name' of a torch.nn.functional activation
            function. All of other key values pairs are bound to the function as a partial.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: The activation function.
    """
    config = deepcopy(ffn_act_fn)
    name = config.pop('name')
    if not hasattr(nn.functional, name):
        raise ValueError(f'Unrecognised activation function name ({name}).')
    act = getattr(nn.functional, name)
    return partial(act, **config)


#############################################################################
# Copied from LLaMaAttention
#############################################################################


def _get_unpad_data(attention_mask: torch.Tensor):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32),
                       (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class DbrxAttention(nn.Module):
    """Multi-head self attention."""

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 max_position_embeddings: int,
                 attn_config: DbrxAttentionConfig,
                 block_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = max_position_embeddings
        self.block_idx = block_idx
        self.config = attn_config
        if block_idx is None:
            logger.warning_once(
                f'Instantiating {self.__class__.__name__} without passing a `block_idx` is not recommended and will '
                +
                'lead to errors during the forward call if caching is used. Please make sure to provide a `block_idx` '
                + 'when creating this class.')

        self.attn_pdrop = attn_config.attn_pdrop
        self.clip_qkv = attn_config.clip_qkv
        self.num_key_value_heads = attn_config.kv_n_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = attn_config.rope_theta

        self.Wqkv = nn.Linear(self.hidden_size,
                              self.hidden_size +
                              2 * self.num_key_value_heads * self.head_dim,
                              bias=False)
        self.out_proj = nn.Linear(self.hidden_size,
                                  self.hidden_size,
                                  bias=False)
        self.rotary_emb = DbrxRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv_states = qkv_states.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        query_states, key_states, value_states = qkv_states.split(
            [
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            dim=2,
        )

        query_states = query_states.view(bsz, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, 'past_key_value', past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states,
                                                        key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {
                'sin': sin,
                'cos': cos,
                'cache_position': cache_position
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.block_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights,
                                             p=self.attn_pdrop,
                                             training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
                + f' {attn_output.size()}')

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DbrxFlashAttention2(DbrxAttention):
    """Dbrx flash attention module.

    This module inherits from `DbrxAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it
    calls the public API of flash attention.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        if not is_flash_attn_2_available():
            raise ImportError(
                'Flash Attention 2 is not available. Please install it with `pip install flash-attn`.'
            )

        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        logger.info(
            'Implicitly setting `output_attentions` to False as it is not supported in Flash Attention.'
        )
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv_states = qkv_states.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        query_states, key_states, value_states = qkv_states.split(
            [
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            dim=2,
        )

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states,
                                                        key_states, cos, sin)

        past_key_value = getattr(self, 'past_key_value', past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                'sin': sin,
                'cos': cos,
                'cache_position': cache_position
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.block_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout
        # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attn_pdrop if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, '_pre_quantization_dtype'):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = query_states.dtype

            logger.warning_once(
                f'The input hidden states seems to be silently casted in float32, this might be '
                +
                f'related to the fact you have upcasted embedding or layer norm layers in '
                + f'float32. We will cast back the input in {target_dtype}.')

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(bsz, q_len,
                                          self.hidden_size).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value  # type: ignore

    def _flash_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Union[torch.LongTensor, None],
        query_length: int,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
    ):
        """Use FlashAttention, stripping padding tokens if necessary.

        Args:
            query_states (torch.Tensor): Input query states to be passed to Flash Attention API
            key_states (torch.Tensor): Input key states to be passed to Flash Attention API
            value_states (torch.Tensor): Input value states to be passed to Flash Attention API
            attention_mask (torch.LongTensor | None): The padding mask - corresponds to a tensor of size
                (batch_size, seq_len) where 0 stands for the position of padding tokens and 1
                for the position of non-padding tokens.
            query_length (int): The length of the query sequence
            dropout (float): Attention dropout
            softmax_scale (float, optional): The scaling of QK^T before applying softmax.
                Defaults to 1 / sqrt(head_dim)
        """
        causal = True
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask,
                query_length)

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

            attn_output = pad_input(
                attn_output_unpad,
                indices_q,
                batch_size,
                query_length,
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(self, query_layer: torch.Tensor, key_layer: torch.Tensor,
                    value_layer: torch.Tensor, attention_mask: torch.Tensor,
                    query_length: int):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
            attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads,
                              head_dim), indices_k)
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads,
                                head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads,
                                    head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


DBRX_ATTENTION_CLASSES = {
    'eager': DbrxAttention,
    'flash_attention_2': DbrxFlashAttention2,
}


class DbrxNormAttentionNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        resid_pdrop: float,
        attn_implementation: str,
        attn_config: DbrxAttentionConfig,
        block_idx: Optional[int] = None,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.resid_pdrop = resid_pdrop
        self.norm_1 = nn.LayerNorm(hidden_size, bias=False)
        self.attn = DBRX_ATTENTION_CLASSES[attn_implementation](
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
            attn_config=attn_config,
            block_idx=block_idx,
        )
        self.norm_2 = nn.LayerNorm(hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[Cache]]:

        residual_states = hidden_states
        hidden_states = self.norm_1(hidden_states).to(hidden_states.dtype)

        hidden_states, attn_weights, past_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.resid_pdrop,
                                              training=self.training)
        hidden_states = hidden_states + residual_states

        residual_states = hidden_states
        hidden_states = self.norm_2(hidden_states).to(hidden_states.dtype)

        return residual_states, hidden_states, attn_weights, past_key_value


class DbrxRouter(nn.Module):

    def __init__(self, hidden_size: int, moe_num_experts: int, moe_top_k: int,
                 moe_jitter_eps: Optional[float],
                 moe_normalize_expert_weights: Optional[float],
                 uniform_expert_assignment: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_normalize_expert_weights = moe_normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment

        self.layer = nn.Linear(self.hidden_size,
                               self.moe_num_experts,
                               bias=False)

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.moe_jitter_eps is None:
            raise RuntimeError('The router does not have moe_jitter_eps set.')
        low = 1.0 - self.moe_jitter_eps
        high = 1.0 + self.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        if self.training and self.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        weights = self.layer(x.view(-1,
                                    x.shape[-1])).softmax(dim=-1,
                                                          dtype=torch.float32)
        top_weights, top_experts = torch.topk(weights, self.moe_top_k, dim=-1)

        if self.moe_normalize_expert_weights:
            top_weights = top_weights / torch.norm(
                top_weights,
                p=self.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True)

        if self.uniform_expert_assignment:
            with torch.no_grad():
                uniform_tensor = torch.arange(
                    0,
                    top_experts.numel(),
                    device=top_experts.device,
                    dtype=top_experts.dtype) % self.moe_num_experts
                top_experts = uniform_tensor.reshape(top_experts.shape)
                # Note, weights and top_weights are not changed

        weights = weights.to(x.dtype)
        top_weights = top_weights.to(x.dtype)
        return weights, top_weights, top_experts  # type: ignore


class DbrxExpertGLU(nn.Module):

    def __init__(self, hidden_size: int, ffn_hidden_size: int,
                 moe_num_experts: int, ffn_act_fn: dict):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts

        self.w1 = nn.Parameter(
            torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.v1 = nn.Parameter(
            torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.w2 = nn.Parameter(
            torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.activation_fn = resolve_ffn_act_fn(ffn_act_fn)

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        expert_w1 = self.w1.view(self.moe_num_experts, self.ffn_hidden_size,
                                 self.hidden_size)[expert_idx]
        expert_v1 = self.v1.view(self.moe_num_experts, self.ffn_hidden_size,
                                 self.hidden_size)[expert_idx]
        expert_w2 = self.w2.view(self.moe_num_experts, self.ffn_hidden_size,
                                 self.hidden_size)[expert_idx]

        x1 = x.matmul(expert_w1.t())
        x2 = x.matmul(expert_v1.t())
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = x1.matmul(expert_w2)
        return x1


class DbrxExperts(nn.Module):

    def __init__(self, hidden_size: int, ffn_hidden_size: int,
                 moe_num_experts: int, ffn_act_fn: dict):
        super().__init__()
        self.moe_num_experts = moe_num_experts
        self.mlp = DbrxExpertGLU(hidden_size=hidden_size,
                                 ffn_hidden_size=ffn_hidden_size,
                                 moe_num_experts=moe_num_experts,
                                 ffn_act_fn=ffn_act_fn)

    def forward(self, x: torch.Tensor, weights: torch.Tensor,
                top_weights: torch.Tensor,
                top_experts: torch.LongTensor) -> torch.Tensor:
        bsz, q_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)

        expert_mask = nn.functional.one_hot(
            top_experts, num_classes=self.moe_num_experts).permute(2, 1, 0)
        for expert_idx in range(0, self.moe_num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue

            token_list = token_idx.tolist()
            topk_list = topk_idx.tolist()

            expert_tokens = x[None, token_list].reshape(-1, hidden_size)
            expert_out = self.mlp(
                expert_tokens, expert_idx) * top_weights[token_list, topk_list,
                                                         None]

            out.index_add_(0, token_idx, expert_out)

        out = out.reshape(bsz, q_len, hidden_size)
        return out


class DbrxFFN(nn.Module):

    def __init__(self, hidden_size: int, ffn_config: DbrxFFNConfig):
        super().__init__()

        self.router = DbrxRouter(
            hidden_size,
            moe_num_experts=ffn_config.moe_num_experts,
            moe_top_k=ffn_config.moe_top_k,
            moe_jitter_eps=ffn_config.moe_jitter_eps,
            moe_normalize_expert_weights=ffn_config.
            moe_normalize_expert_weights,
            uniform_expert_assignment=ffn_config.uniform_expert_assignment,
        )

        self.experts = DbrxExperts(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_config.ffn_hidden_size,
            moe_num_experts=ffn_config.moe_num_experts,
            ffn_act_fn=ffn_config.ffn_act_fn,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights, top_weights, top_experts = self.router(x)
        out = self.experts(x, weights, top_weights, top_experts)
        return out, weights


class DbrxBlock(nn.Module):

    def __init__(self, config: DbrxConfig, block_idx: int):
        super().__init__()
        self.hidden_size = config.d_model
        self.resid_pdrop = config.resid_pdrop
        self.block_idx = block_idx
        self.norm_attn_norm = DbrxNormAttentionNorm(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            max_position_embeddings=config.max_seq_len,
            resid_pdrop=config.resid_pdrop,
            attn_implementation=config._attn_implementation,
            attn_config=config.attn_config,
            block_idx=block_idx,
        )
        self.ffn = DbrxFFN(hidden_size=config.d_model,
                           ffn_config=config.ffn_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]],
               Tuple[torch.Tensor, Optional[Cache]], Tuple[
                   torch.Tensor, Optional[torch.Tensor], Optional[Cache]],
               Tuple[torch.Tensor, Optional[torch.Tensor],
                     Optional[torch.Tensor]], Tuple[
                         torch.Tensor, Optional[Cache], Optional[torch.Tensor]],
               Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache],
                     Optional[torch.Tensor]],]:
        """Forward function for DbrxBlock.

        Args:
            hidden_states (`torch.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            position_ids (`torch.LongTensor`): position ids of shape `(batch, seq_len)`
            attention_mask (`torch.Tensor`, optional): attention mask of size (batch_size, sequence_length)
                if flash attention is used or (batch_size, 1, query_sequence_length, key_sequence_length)
                if default attention is used.
            past_key_value (`Tuple(torch.Tensor)`, optional): cached past key and value projection states
            output_attentions (`bool`, optional): Whether or not to return the attentions tensors of all
                attention layers. See `attentions` under returned tensors for more detail.
            output_router_logits (`bool`, optional): Whether or not to return the router logits.
            use_cache (`bool`, optional): If set to `True`, `past_key_values` key value states are
                returned and can be used to speed up decoding (see `past_key_values`).
            cache_position (`torch.LongTensor`, optional): position ids of the cache
        """
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
            )

        # Norm + Attention + Norm
        resid_states, hidden_states, self_attn_weights, present_key_value = self.norm_attn_norm(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        # Fully Connected
        hidden_states, router_logits = self.ffn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.resid_pdrop,
                                              training=self.training)
        hidden_states = resid_states + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class DbrxPreTrainedModel(PreTrainedModel):
    config_class = DbrxConfig
    base_model_prefix = 'transformer'
    supports_gradient_checkpointing = True
    _no_split_modules = ['DbrxBlock']
    _skip_keys_device_placement = ['past_key_values']
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, DbrxExpertGLU):
            module.w1.data.normal_(mean=0.0, std=std)
            module.v1.data.normal_(mean=0.0, std=std)
            module.w2.data.normal_(mean=0.0, std=std)

    def _setup_cache(self, cache_cls: Any, max_batch_size: int,
                     max_cache_len: int):  # TODO: how to set var type of class?
        if self.config._attn_implementation == 'flash_attention_2' and cache_cls == StaticCache:
            raise ValueError(
                '`static` cache implementation is not compatible with ' +
                '`attn_implementation==flash_attention_2`. Make sure to use ' +
                '`spda` in the mean time and open an issue at https://github.com/huggingface/transformers.'
            )

        for block in self.transformer.blocks:
            device = block.norm_attn_norm.norm_1.weight.device
            if hasattr(self.config, '_pre_quantization_dtype'):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = block.norm_attn_norm.attn.out_proj.weight.dtype
            block.norm_attn_norm.attn.past_key_value = cache_cls(self.config,
                                                                 max_batch_size,
                                                                 max_cache_len,
                                                                 device=device,
                                                                 dtype=dtype)

    def _reset_cache(self):
        for block in self.transformer.blocks:
            block.norm_attn_norm.attn.past_key_value = None


class DbrxModel(DbrxPreTrainedModel):
    """Transformer decoder consisting of *config.num_hidden_layers*

    [`DbrxBlock`] layers.

    Args:
        config: DbrxConfig
    """

    def __init__(self, config: DbrxConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.emb_pdrop = config.emb_pdrop

        self.wte = nn.Embedding(config.vocab_size, config.d_model,
                                self.padding_idx)
        self.blocks = nn.ModuleList([
            DbrxBlock(config, block_idx) for block_idx in range(config.n_layers)
        ])
        self.norm_f = nn.LayerNorm(config.d_model, bias=False)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, value: nn.Embedding):
        self.wte = value

    def _autocast_input_embeddings(self,
                                   inputs_embeds: torch.Tensor) -> torch.Tensor:
        if inputs_embeds.device.type == 'cuda' and torch.is_autocast_enabled():
            return inputs_embeds.to(dtype=torch.get_autocast_gpu_dtype())
        elif inputs_embeds.device.type == 'cpu' and torch.is_autocast_cpu_enabled(
        ):
            return inputs_embeds.to(dtype=torch.get_autocast_cpu_dtype())
        else:
            return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        output_router_logits = (output_router_logits
                                if output_router_logits is not None else
                                self.config.output_router_logits)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one'
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.'
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        inputs_embeds = self._autocast_input_embeddings(
            inputs_embeds)  # type: ignore
        inputs_embeds = nn.functional.dropout(inputs_embeds,
                                              p=self.emb_pdrop,
                                              training=self.training)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values)
                past_seen_tokens = past_key_values.get_seq_length(  # type: ignore
                )

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError(
                    'cache_position is a required argument when using StaticCache.'
                )
            cache_position = torch.arange(  # type: ignore
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)  # type: ignore

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds,
                                               cache_position)  # type: ignore

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore

            if self.gradient_checkpointing and self.training:
                block_outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            else:
                block_outputs = block(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = block_outputs[0]

            if use_cache:
                next_decoder_cache = block_outputs[
                    2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (block_outputs[1],)  # type: ignore

            if output_router_logits:
                all_router_logits += (block_outputs[-1],)  # type: ignore

        hidden_states = self.norm_f(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # type: ignore

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()  # type: ignore
                if isinstance(next_decoder_cache, Cache) else
                next_decoder_cache)
        if not return_dict:
            return tuple(v for v in [
                hidden_states, next_cache, all_hidden_states, all_self_attns,
                all_router_logits
            ] if v is not None)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(
            self, attention_mask: Optional[torch.Tensor],
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor) -> Optional[torch.Tensor]:
        if self.config._attn_implementation == 'flash_attention_2':
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(self.blocks[0].norm_attn_norm.attn,
                   'past_key_value'):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (attention_mask.shape[-1] if isinstance(
                attention_mask, torch.Tensor) else cache_position[-1] + 1)
        target_length = int(target_length)

        causal_mask = torch.full((sequence_length, target_length),
                                 fill_value=min_dtype,
                                 dtype=dtype,
                                 device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None,
                                  None, :, :].expand(input_tensor.shape[0], 1,
                                                     -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone(
            )  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(
                    0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[
                    ..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[
                        -2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(
                    dtype=dtype) * min_dtype
                causal_mask[:mask_shape[0], :mask_shape[1],
                            offset:mask_shape[2] +
                            offset, :mask_shape[3]] = mask_slice

        if (self.config._attn_implementation == 'sdpa' and
                attention_mask is not None and
                attention_mask.device.type == 'cuda'):
            # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing() or
                isinstance(input_tensor, torch.fx.Proxy) or  # type: ignore
                (hasattr(torch, '_dynamo') and torch._dynamo.is_compiling()))
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(
                    causal_mask, min_dtype)

        return causal_mask


class DbrxForCausalLM(DbrxPreTrainedModel):

    def __init__(self, config: DbrxConfig):
        super().__init__(config)
        self.transformer = DbrxModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.ffn_config.moe_num_experts
        self.num_experts_per_tok = config.ffn_config.moe_top_k

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding):
        self.transformer.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: DbrxModel):
        self.transformer = decoder

    def get_decoder(self) -> DbrxModel:
        return self.transformer

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""Forward function for causal language modeling.

        Example:
        ```python
        >>> from transformers import AutoTokenizer, DbrxForCausalLM

        >>> model = DbrxForCausalLM.from_pretrained("databricks/dbrx")
        >>> tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        output_router_logits = (output_router_logits
                                if output_router_logits is not None else
                                self.config.output_router_logits)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None and loss is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(
                    loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.Tensor,
            past_key_values: Optional[Cache] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs: Any) -> Dict[str, Any]:
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[
                    1] > input_ids.shape[1]:
                input_ids = input_ids[:,
                                      -(attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (max_cache_length is not None and attention_mask is not None and
                    cache_length + input_ids.shape[1] > max_cache_length):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if self.generation_config.cache_implementation == 'static':
            # generation with static cache
            cache_position = kwargs.get('cache_position', None)
            if cache_position is None:
                past_length = 0
            else:
                past_length = cache_position[-1] + 1
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:,
                                        past_length:] if position_ids is not None else None

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        input_length = position_ids.shape[
            -1] if position_ids is not None else input_ids.shape[-1]
        cache_position = torch.arange(past_length,
                                      past_length + input_length,
                                      device=input_ids.device)
        position_ids = position_ids.contiguous(
        ) if position_ids is not None else None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        model_inputs.update(
            { # type: ignore
                'position_ids': position_ids,
                'cache_position': cache_position,
                'past_key_values': past_key_values,
                'use_cache': kwargs.get('use_cache'),
                'attention_mask': attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values: Cache, beam_idx: torch.LongTensor):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past),)
        return reordered_past

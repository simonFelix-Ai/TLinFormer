####################################################################################
# Copyright (c) 2025, Zhongpan Tang
#
# Licensed under the Academic and Non-Commercial Research License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   LICENSE.md file in the repository
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For commercial use, please contact: tangzhongp@gmail.com
####################################################################################

# pylint: disable=too-many-positional-arguments
"""
This file
"""
import logging
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# pylint: disable=import-error
from src.models.position_encoding import HybridRotaryEmbedding
from src.models.cache_mixin import CacheMixin


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN layer."""
    def __init__(self, n_embd: int, dropout: float,
                 hidden_dim_multiplier: float = 2/3):
        super().__init__()

        # According to the Llama paper, the hidden layer dimension is typically 2/3 * 4 * d
        # and ensure it's divisible by 256 for better hardware efficiency
        hidden_dim = int(hidden_dim_multiplier * 4 * n_embd)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False) # Gating layer
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Mark as residual output
        self.w2._is_residual_output = True # pylint: disable=protected-access

    def forward(self, x):
        """Forward pass."""
        # SiLU(x) = x * sigmoid(x)
        # F.silu is a highly efficient built-in implementation in PyTorch
        silu_out = F.silu(self.w1(x))
        gate_out = self.w3(x)

        # Element-wise multiplication
        x = silu_out * gate_out

        # Output projection and dropout
        x = self.w2(x)
        x = self.dropout(x)
        return x


class QkvGetBlock(nn.Module, CacheMixin):
    """QKV get block."""
    def __init__(self, n_embd, n_head, share_status):
        super().__init__()
        self._init_cache_state(share_status)
        # logging.info("unique_id: %s", self.unique_id)

        self.n_head = n_head
        self.head_dim = n_embd // n_head

    # pylint: disable=too-many-branches, too-many-locals
    def forward(self, normed_x_proj, cache_key="x_qkv"):
        """Forward pass for QkvGetBlock."""
        if normed_x_proj is None:
            return None, None, None

        b, l_x, _ = normed_x_proj.shape

        cat_flag = False
        gen_flag = True
        l_cal = l_x
        x = normed_x_proj

        if l_cal > 0:
            if self.is_cache_on():
                if cache_key == "x_qkv":
                    gen_flag = True
                    if self.is_cache_valid(cache_key):
                        x = x[:, -1:, :]
                        l_cal = 1
                        cat_flag = True
        
            if gen_flag:
                q, k, v = x.chunk(3, dim=-1) # (B, L, 3 * n_embd)
                q = q.view(b, l_cal, self.n_head, self.head_dim).transpose(1, 2)
                k = k.view(b, l_cal, self.n_head, self.head_dim).transpose(1, 2)
                v = v.view(b, l_cal, self.n_head, self.head_dim).transpose(1, 2)

            if cat_flag:
                past_q, past_k, past_v = self.get_unique_cache(cache_key)
                # bacause x_qkv is so small, so the torch.cat don't take many time
                q = torch.cat([past_q, q], dim=2)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            if self.is_cache_on():
                self.set_unique_cache(cache_key, (q, k, v))
        else:
            q, k, v = self.get_unique_cache(cache_key)

        return q, k, v


class TransformerBlock(nn.Module, CacheMixin):
    """Transformer block."""
    # R0902: Too many instance attributes
    # R0913: Too many arguments
    # R0917: Too many positional arguments
    def __init__(self, n_embd, n_head, dropout, rope, share_status):
        super().__init__()
        self._init_cache_state(share_status)

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        self.rope = rope
        self.get_qkv_x = QkvGetBlock(n_embd, n_head, share_status)
        #"减少 GPU 核函数调用次数"
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        # 简单地给这个子模块添加一个特殊的属性, _init_weights 方法中使用
        self.qkv_proj._is_residual_output = True # pylint: disable=protected-access

        self.out_proj = nn.Linear(n_embd, n_embd)
        # 简单地给这个子模块添加一个特殊的属性, _init_weights 方法中使用
        self.out_proj._is_residual_output = True # pylint: disable=protected-access

        self.ffn = SwiGLUFFN(n_embd, dropout)

        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        q, k, v  = self.get_qkv_x(self.qkv_proj(self.norm1(x)), "x_qkv")

        q, k = self.rope(q, k, 0, 0)

        # 使用 is_causal=True，不再需要 attn_mask！
        # pylint: disable=not-callable
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(x.shape[0], x.shape[1], self.n_embd)
        x = x + self.out_proj(attn_out)
        x = x + self.ffn(self.norm2(x))

        return x

# ======================================================================================
# SelfAttentionTransformer
# ======================================================================================
class SelfAttentionTransformer(nn.Module):
    """Self attention transformer."""
    # R0913: Too many arguments
    # R0917: Too many positional arguments
    def __init__(self, n_embd, n_head, n_transformer_block, vocab_size, dropout, share_status):
        super().__init__()

        self.head_dim = n_embd // n_head
        self.config = share_status['config']

        rope_precompute_threshold = share_status['config']['model']['model_args'].get(
            'rope_precompute_threshold')
        share_rope = HybridRotaryEmbedding(
            dim=self.head_dim, precompute_threshold=rope_precompute_threshold)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                n_embd, n_head, dropout, share_rope, share_status
            ) for _ in range(n_transformer_block)
        ])

        self.norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        for layer in self.transformer_layers:
            use_checkpoint = self.config['training'].get('use_checkpoint')
            if use_checkpoint:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.norm(x)
        x = self.lm_head(x)

        return x


class BaseLLM(nn.Module, CacheMixin):
    """Base LLM."""
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.share_status = self.create_share_status(config)

        model_arg_config = config['model']['model_args']

        vocab_size = model_arg_config.get('vocab_size')

        n_head = model_arg_config.get('n_head')
        n_embd = model_arg_config.get('n_embd')
        dropout = model_arg_config.get('dropout')
        n_transformer_block = model_arg_config.get('n_transformer_block')

        if n_embd % n_head != 0:
            raise ValueError(f"n_embd({n_embd}) % n_head({n_head}) != 0")

        if (n_embd // n_head) % 2 != 0:
            raise ValueError(f"sub head dim ({n_embd // n_head}) must be even")

        self.total_layers = n_transformer_block
        self.generate_window_size = config['data']['max_train_seq_len']

        self.embedding = nn.Embedding(vocab_size, n_embd)

        self.transformer = SelfAttentionTransformer(
            n_embd, n_head,
            n_transformer_block,
            vocab_size,
            dropout,
            self.share_status
        )

        # ================================================================= #
        #  ★★★ 在这里添加权重共享代码 ★★★
        # ================================================================= #
        self.transformer.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initializes weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

            # --- 使用标记来识别，而不是名字 ---
            # pylint: disable=protected-access
            if hasattr(module, '_is_residual_output') and module._is_residual_output:
                # logging.info("Scaling weights for residual output layer: %s", module)
                torch.nn.init.normal_(
                    module.weight, mean=0.0,
                    std=0.02 / math.sqrt(2 * self.total_layers))

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def train_llm(self, x):
        """Trains the LLM."""
        self.set_cache_on(self.config['training'].get('kv_cache_on'))
        self.share_status["prev_context_len"] = 0
        self.share_status["context_changed"] = True
        self.clean_all_cache()

        logits = self.transformer(x)

        self.clean_all_cache()
        self.set_cache_on(False)

        return logits

    def inference(self, x, cache_on):
        """Inference."""
        self.set_cache_on(cache_on)

        logits = self.transformer(x)

        return logits

    # pylint: disable=too-many-arguments
    def forward(self, x, cache_on=False, evaluate_mode=False):
        """Forward pass for the LLM."""
        x = self.embedding(x)

        if self.training or evaluate_mode:
            all_logits = self.train_llm(x)
        else:
            all_logits = self.inference(x, cache_on)

        # self.dump_all_cache_keys("forward end")

        return all_logits

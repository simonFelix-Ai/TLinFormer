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
        l_cal = l_x
        x = normed_x_proj

        context_changed = self.share_status['context_changed']
        cache_on = self.is_cache_on()
        past_cache_valid = self.is_cache_valid(cache_key)

        gen_flag = True
        if ( cache_key in ("context_origin_qkv", "context_zip_qkv") and
            not context_changed and past_cache_valid and cache_on
        ):
            gen_flag = False

        cat_flag  = False
        if ((cache_key == "x_qkv")
        and cache_on and past_cache_valid and not context_changed and gen_flag
        ):
            cat_flag  = True
            l_cal = 1
            x = x[:, -l_cal:, :]
        
        if gen_flag:
            q, k, v = x.chunk(3, dim=-1)
            q = q.view(b, l_cal, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(b, l_cal, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(b, l_cal, self.n_head, self.head_dim).transpose(1, 2)

            if cat_flag:
                past_q, past_k, past_v = self.get_unique_cache(cache_key)
                # bacause x_qkv is so small, so the torch.cat don't take many time
                q = torch.cat([past_q, q], dim=2)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            if cache_on:
                save_q = None if cache_key == "context_origin_qkv" else q
                self.set_unique_cache(cache_key, (save_q, k, v))
        else:
            q, k, v = self.get_unique_cache(cache_key)

        return q, k, v

# pylint: disable=too-many-instance-attributes
class TransformerBlockInnerLayer(nn.Module, CacheMixin):
    """Inner layer of the transformer block."""
    # pylint: disable=too-many-arguments
    def __init__(self, n_embd, n_head, dropout, rope, share_status, 
                 layer_type="compress"):
        super().__init__()
        self._init_cache_state(share_status)

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        self.layer_type = layer_type
        self.rope = rope

        self.get_qkv_x = QkvGetBlock(n_embd, n_head, share_status)
        self.get_qkv_context_zip = QkvGetBlock(n_embd, n_head, share_status)
        self.get_qkv_context_origin = QkvGetBlock(n_embd, n_head, share_status)

        # "Reduce GPU kernel function calls"
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        # Simply add a special attribute to this submodule for use in the _init_weights method
        self.qkv_proj._is_residual_output = True # pylint: disable=protected-access

        self.out_proj = nn.Linear(n_embd, n_embd)
        # Simply add a special attribute to this submodule for use in the _init_weights method
        self.out_proj._is_residual_output = True # pylint: disable=protected-access

        self.ffn = SwiGLUFFN(n_embd, dropout)

        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def context_select(self, context_origin, history_window_start_pos):
        """Selects context based on the observation window."""
        if context_origin is None:
            return None
        
        selected_context = context_origin[:, history_window_start_pos:, :]

        return selected_context
        
    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def forward(self, context_origin, context_zip, x, 
                origin_context_pos, history_window_start_pos, x_pos, 
                context_origin_qkv):
        """Forward pass for the inner layer."""
        b, l_x, _ = x.shape

        x_save = x
        # context_origin_save = context_origin
        context_zip_save = context_zip

        attn_cal_list = []

        # logging.info("proc layer: %s, %s", self.layer_type, l_x)
        
        #######################################################
        ## gen q k v
        layer_qkv_gen_flag = {
            # (x_qkv_gen_flag, context_origin_qkv_gen_flag, context_zip_qkv_gen_flag)
            "compress": (True, True, False),
            "hidden":  (True, False, True),
            "restore": (True, False, True)
        }
        
        (x_qkv_gen_flag, 
        context_origin_qkv_gen_flag,
        context_zip_qkv_gen_flag) = layer_qkv_gen_flag[self.layer_type]

        q_x, k_x, v_x = None, None, None
        q_context_zip, k_context_zip, v_context_zip = None, None, None
        if x_qkv_gen_flag:
            q_x, k_x, v_x  = self.get_qkv_x(
                self.qkv_proj(self.norm1(x)), "x_qkv")
        
        fake_val = 0xBAD * torch.ones_like(x)        
        if context_zip_qkv_gen_flag and context_zip is not None:
            if self.is_cache_valid("history_finish_flag"):
                qkv_context_zip = self.get_qkv_context_zip(fake_val, "context_zip_qkv")
            else:
                norm_prj = self.qkv_proj(self.norm1(context_zip))
                qkv_context_zip = self.get_qkv_context_zip(norm_prj, "context_zip_qkv")
            q_context_zip, k_context_zip, v_context_zip = qkv_context_zip

        if context_origin_qkv_gen_flag and context_origin is not None:
            if self.is_cache_valid("history_finish_flag"):
                context_origin_qkv = self.get_qkv_context_origin(fake_val, "context_origin_qkv")
            else:
                norm_prj = self.qkv_proj(self.norm1(context_origin))
                context_origin_qkv = self.get_qkv_context_origin(norm_prj, "context_origin_qkv")
        if context_origin_qkv is not None:  
            q_context_origin, k_context_origin, v_context_origin = context_origin_qkv
        else:
            q_context_origin, k_context_origin, v_context_origin = None, None, None
        #######################################################
        # for context_origin, context_zip select
        context_selected = None
        q_selected_context = None
        if self.layer_type == "compress": 
            if not self.is_cache_valid("history_finish_flag"):
                if context_origin is not None:
                    context_selected = self.context_select(context_origin, history_window_start_pos)
                    q_selected_context = q_context_origin[
                        : , :, history_window_start_pos:, :]
        
        layer_qkv_info = {
            # for_c: (q_context, k_context, v_context, context_q_pos_start,
            # context_kv_pos_start) # Context information for attention calculation
            # for_x: (k_context, v_context, context_kv_pos_start)
            # # Context information for cross-attention calculation
            "compress": {
                "for_c": (q_selected_context, k_context_origin, v_context_origin,
                        history_window_start_pos, origin_context_pos),
                "for_x": (k_context_origin, v_context_origin, origin_context_pos)
            },
            "hidden":  {
                "for_c": (q_context_zip, k_context_zip, v_context_zip,
                        history_window_start_pos, history_window_start_pos),
                "for_x": (k_context_zip, v_context_zip, history_window_start_pos)
            },
            "restore": {
                "for_c": (q_context_origin, k_context_zip, v_context_zip,
                        origin_context_pos, history_window_start_pos),
                "for_x": (k_context_zip, v_context_zip, history_window_start_pos)
            }
        }
        if self.is_cache_valid("history_finish_flag"):
            for key in layer_qkv_info.keys():
                layer_qkv_info[key]['for_c'] = (None, None, None, None, None)
            
        ####################################################### 
        # ### for x atten cal
        k_context, v_context, context_kv_pos_start = layer_qkv_info[self.layer_type]["for_x"]
        if context_origin is not None and k_context is not None: # for context
            qx_cal_len, is_causal, cache_key = l_x, False, "x_context_attn"

            if (
                self.is_cache_on()
                and self.is_cache_valid(cache_key)
            ):
                qx_cal_len = 1

            attn_cal_list.append((
                "for_x", cache_key, is_causal,
                q_x[:,:,-qx_cal_len:,:], k_context, v_context,
                x_pos+l_x-qx_cal_len, context_kv_pos_start
            ))

        is_causal = True
        attn_cal_list.append((
            "for_x", None, is_causal,
            q_x, k_x, v_x,
            x_pos, x_pos
        ))
        #######################################################

        #######################################################
        # for context attention calculation
        if context_origin is not None:
            (q_context, k_context, v_context, 
             context_q_pos_start, 
             context_kv_pos_start) = layer_qkv_info[self.layer_type]["for_c"]

            if k_context is not None:
                cache_key, is_causal = None, False
                attn_cal_list.append((
                    "for_c", cache_key, is_causal,
                    q_context, k_context, v_context,
                    context_q_pos_start, context_kv_pos_start
                ))
        #######################################################

        #######################################################
        ###### Attention calculation
        #######################################################
        x_attn_out, context_attn_out = None, None
        for attn_cal_info in attn_cal_list:
            for_which, cache_key, is_causal, q, k, v, q_pos, k_pos = attn_cal_info
            
            flag_do_cal = True
            if ((for_which == "for_c")
                and self.is_cache_on()
                and self.is_cache_valid(cache_key)
            ):
                flag_do_cal = False
            
            if flag_do_cal:
                q, k = self.rope(q, k, q_pos, k_pos)
                # pylint: disable=not-callable
                attn_out =  F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal
                ).permute(0, 2, 1, 3).contiguous().view(b, q.shape[-2], self.n_embd)
            else:
                attn_out = self.get_unique_cache(cache_key)

            # logging.info(f"{layer_type} {attn_out.size()} l_x {l_x}")
            if ((for_which == "for_x") 
                and self.is_cache_on()
                and (l_x != attn_out.shape[1])
                and self.is_cache_valid(cache_key)):
                    attn_out = torch.cat(
                        [self.get_unique_cache(cache_key), attn_out], dim=1)

            if (
                self.is_cache_on()
                and (cache_key is not None)
            ):
                self.set_unique_cache(cache_key, attn_out)

            if for_which == "for_x":
                if x_attn_out is None:
                    x_attn_out = attn_out
                else:
                    x_attn_out = x_attn_out + attn_out
            else:
                if context_attn_out is None:
                    context_attn_out = attn_out
                else:
                    context_attn_out = context_attn_out + attn_out

        #####################################################################################
        # Perform ffn processing. Since ffn does not fuse L-dimensional information,
        # we can fuse first, then calculate, then split
        #####################################################################################
        # logging.info(f"{layer_type}: {L_x}  {joint_process_input.size()} {joint_process_attn_out.size()}")
        cache_key = "history_finish_flag"
        if self.is_cache_valid(cache_key):
            # just a bad valude, don't need to care
            context_process_out = self.get_unique_cache(cache_key)
        elif context_attn_out is not None:
            context_layer_residual_in_info = {
                # for_c: (context) #
                "compress": (context_selected),
                "hidden":  (context_zip_save),
                "restore": (context_origin)
            }
            context_process_input = context_layer_residual_in_info[self.layer_type]
            context_process_out = context_process_input + self.out_proj(context_attn_out) 
            context_process_out = context_process_out + self.ffn(self.norm2(context_process_out))
            if self.is_cache_on():
                # just make "history_finish_flag" flag to valid, use a fake value to store
                # self.set_unique_cache(cache_key, 0xBAD * torch.ones_like(x_save))
                self.set_unique_cache(cache_key, True)
        else:
            context_process_out = None

        x_process_out = x_save + self.out_proj(x_attn_out) 
        x_process_out = x_process_out + self.ffn(self.norm2(x_process_out))

        context_tranformed = context_process_out
        x_tranformed = x_process_out

        return context_tranformed, x_tranformed, context_origin_qkv

class TransformerBlock(nn.Module):
    """Transformer block."""
    # pylint: disable=too-many-arguments
    def __init__(self, n_embd, n_head, dropout, rope, share_status):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.compress_context_layer = TransformerBlockInnerLayer(
            n_embd, n_head, dropout, rope, share_status, layer_type="compress")

        block_inner_depth = share_status['config']['model']['model_args'].get(
            'block_inner_depth')
        hidden_layer_depth = block_inner_depth - 2
        if hidden_layer_depth < 0:
            raise ValueError(
                f"block_inner_depth must > 1, current is {block_inner_depth}")

        self.hidden_layers = nn.ModuleList([
            TransformerBlockInnerLayer(
                n_embd, n_head, dropout, rope, share_status, layer_type="hidden"
            ) for _ in range(hidden_layer_depth)
        ])

        self.restore_context_shape_layer = TransformerBlockInnerLayer(
            n_embd, n_head, dropout, rope, share_status, layer_type="restore")


    # pylint: disable=arguments-differ
    def forward(self, context, x, h_pos, x_pos): # pylint: disable=arguments-differ
        """Forward pass for the transformer block."""
        if context is None:
            origin_context_pos = None
        else:
            origin_context_pos = 0
        
        context_origin = context
        context_origin_qkv = None    
        context_zip = None
        
        context_zip, x, context_origin_qkv = self.compress_context_layer(
            context_origin, context_zip, x,
            origin_context_pos, h_pos, x_pos,
            context_origin_qkv
        )

        for hidden_layer in self.hidden_layers:
            context_zip, x, _ = hidden_layer(
                context_origin, context_zip, x,
                origin_context_pos, h_pos, x_pos,
                context_origin_qkv
            )

        context_shape_restored, x, _ = self.restore_context_shape_layer(
            context_origin, context_zip, x, 
            origin_context_pos, h_pos, x_pos,
            context_origin_qkv
        )

        return context_shape_restored, x

# ======================================================================================
# Tang Linear Attention (TLT) : A Lossless Approach to Linearizing Transformer Attention
# ======================================================================================
class TLinFormer(nn.Module):
    """Tang Linear Transformer."""
    # pylint: disable=too-many-arguments
    def __init__(self, n_embd, n_head, n_transformer_block, vocab_size, dropout, share_status):
        super().__init__()

        self.head_dim = n_embd // n_head
        self.config = share_status['config']

        self.share_status = share_status
        
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

    def get_history_window_start_pos(self, context_origin):
        """Selects context based on the observation window."""
        if context_origin is None:
            return None

        l_context_origin = context_origin.shape[1]
        
        model_arg_config = self.share_status['config']['model']['model_args']
        history_windown_size = model_arg_config.get('history_windown_size')
        if history_windown_size == "auto":
            # based on: Research on information theory and compressed sensing 
            # (such as the n > C log N rule), C~= 8.33, I just like 12 here, no reason
            # here we still pad to 
            history_windown_size = 12 * math.log(l_context_origin)
            history_windown_size = (history_windown_size + 7) // 8 * 8
        
        history_windown_size = int(history_windown_size)
        
        # Check for some edge cases
        if history_windown_size >= l_context_origin:
            # If the number to be selected is greater than or equal to the context length,
            # return the original context directly
            history_window_start_pos = 0
        else:
            history_window_start_pos = l_context_origin - history_windown_size
        
        return history_window_start_pos

    # pylint: disable=arguments-differ
    def forward(self, context, x) -> torch.Tensor:
        """Forward pass for the Tang Linear Transformer."""
        hspos = self.get_history_window_start_pos(context)
        xpos = 0 if context is None else context.shape[1]
        
        for layer in self.transformer_layers:
            use_checkpoint = self.config['training'].get('use_checkpoint')
            if use_checkpoint:
                context, x = checkpoint(layer, context, x, hspos, xpos, use_reentrant=False)
            else:
                context, x = layer(context, x, hspos, xpos)

        x = self.norm(x)
        x = self.lm_head(x)

        return x

# ======================================================================================
# TOP-LEVEL MODEL: Long Context Language Model
# ======================================================================================
# pylint: disable=too-many-instance-attributes
class TLinLLM(nn.Module, CacheMixin):
    """Tang Linear Long Language Model."""
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

        if (
            (n_embd // n_head) % 2 != 0):
            raise ValueError(
                f"sub head dim ({n_embd // n_head}) must be even")

        block_inner_depth = model_arg_config.get('block_inner_depth')
        self.total_layers = n_transformer_block * block_inner_depth

        self.generate_window_size = model_arg_config.get(
            'generate_window_size')

        self.embedding = nn.Embedding(vocab_size, n_embd)

        self.transformer = TLinFormer(
            n_embd, n_head,
            n_transformer_block,
            vocab_size,
            dropout,
            self.share_status
        )

        # ================================================================= #
        #  ★★★ Add weight sharing code here ★★★
        # ================================================================= #
        self.transformer.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initializes weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

            # --- Use a flag for identification, not the name ---
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
        """Training loop for the LLM."""
        _, l_x, _ = x.shape

        self.set_cache_on(self.config['training'].get('kv_cache_on'))
        self.share_status["prev_context_len"] = 0
        self.share_status["context_changed"] = True
        self.clean_all_cache()

        all_logits = []
        
        for chunk_start_idx in range(0, l_x, self.generate_window_size):
            context = x[:, :chunk_start_idx, :]
            x_chunk = x[:, chunk_start_idx:chunk_start_idx + self.generate_window_size, :]

            l_context = context.shape[1]
            context = None if l_context == 0 else context
            
            self.share_status["context_changed"] = (
                l_context != self.share_status["prev_context_len"])
            if self.share_status["context_changed"]:
                self.clean_invalid_cache()

            logits = self.transformer(context, x_chunk)
            all_logits.append(logits)

            self.share_status["prev_context_len"] = l_context

        full_logits = torch.cat(all_logits, dim=1)

        self.clean_all_cache()
        self.set_cache_on(False)

        return full_logits

    def inference(self, x, cache_on):
        """Inference for the LLM."""
        l_x_remain = x.shape[1] % self.generate_window_size
        if l_x_remain == 0:
            l_x_remain = self.generate_window_size
        l_context = x.shape[1] - l_x_remain
        context = x[:, :l_context, :]
        x_remain = x[:, -l_x_remain:, :]
        context = None if l_context == 0 else context
                
        self.set_cache_on(cache_on)

        self.share_status["context_changed"] = (
            l_context != self.share_status["prev_context_len"])
        if self.share_status["context_changed"]:
            self.clean_invalid_cache()

        logits = self.transformer(context, x_remain)

        self.share_status["prev_context_len"] = l_context

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

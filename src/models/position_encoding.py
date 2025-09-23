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

import torch
import torch.nn as nn
import math
from typing import Tuple, Dict, List
import logging

class SinCosPosEncoding(nn.Module):
    """
    Standard sin/cos absolute position encoding implementation.
    From "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the positional encoding module.

        Args:
        d_model (int): The dimension of the model, i.e., the dimension of the word embeddings.
        dropout (float): The dropout rate.
        max_len (int): The maximum pre-computed sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Check if d_model is even, as positional encoding is calculated in pairs.
        if d_model % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with "
                f"odd d_model: {d_model}"
            )

        # Create a positional encoding matrix of shape (max_len, d_model) that is large enough.
        pe = torch.zeros(max_len, d_model)

        # Create a position tensor of shape (max_len, 1).
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term `1 / (10000^(2i/d_model))`.
        # The shape of div_term is (d_model / 2).
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Use broadcasting to calculate sin and cos.
        # The shape of position * div_term is (max_len, d_model / 2).
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        # Register pe as a buffer.
        # A buffer is part of the model and will be saved in the state_dict, 
        # but it is not considered a model parameter (i.e., it won't be updated by the optimizer).
        # unsqueeze(0) is for convenient addition with batched inputs later. The shape of pe becomes (1, max_len, d_model).
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation function.

        Args:
        x (torch.Tensor): The input word embedding tensor, with shape (batch_size, seq_len, d_model).

        Returns:
        torch.Tensor: The output tensor with positional encoding added, having the same shape as the input.
        """
        # x.size(1) is the sequence length (seq_len).
        # The shape of self.pe is (1, max_len, d_model).
        # self.pe[:, :x.size(1)] slices the positional encoding to the required length, with shape (1, seq_len, d_model).
        # Then it is added to x, utilizing PyTorch's broadcasting mechanism.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class HybridRotaryEmbedding(nn.Module):
    """
    An optimized, true hybrid-strategy RoPE implementation.

    It accurately handles sequences that cross the pre-computation threshold by concatenating the cached part and
    the dynamically computed part, thus avoiding any unnecessary re-computation.
    """
    def __init__(self, dim: int, precompute_threshold: int = 32*1024, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.precompute_threshold = precompute_threshold

        # Pre-store inv_freq only for the dynamic computation part.
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute only up to the threshold length.
        logging.info(f"Optimized RoPE: Pre-computing frequencies up to position {self.precompute_threshold}...")
        t = torch.arange(self.precompute_threshold).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        freqs_cis = torch.polar(torch.ones_like(emb), emb).unsqueeze(0).unsqueeze(0)
        self.register_buffer("freqs_cis_precomputed", freqs_cis, persistent=False)

    def _compute_dynamic_freqs_cis(self, seq_len: int, seq_start: int, device: torch.device) -> torch.Tensor:
        """
        Helper function: responsible only for dynamic computation.
        
        1. dtype parameter to receive the model's main data type (e.g., bfloat16).
        2. Force the use of float32 during computation to ensure numerical stability and operator compatibility.
        3. Convert the result back to the original dtype at the end.
        """
        # --- Perform calculations in a high-precision (float32) environment ---
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            # 1. Create t of type float32.
            t = torch.arange(seq_start, seq_start + seq_len, device=device, dtype=torch.float32)
            
            # 2. inv_freq is already float32, use it directly.
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            
            # 3. emb is also float32.
            emb = torch.cat((freqs, freqs), dim=-1)

            # 4. torch.polar works well on float32.
            freqs_cis_float32 = torch.polar(torch.ones_like(emb), emb).unsqueeze(0).unsqueeze(0)
        
        # --- Convert the final result back to the model's required data type ---
        return freqs_cis_float32
    
    def _apply_rotary_emb(self, x: torch.Tensor, seq_start: int) -> torch.Tensor:
        seq_len = x.shape[-2]
        original_dtype = x.dtype # e.g., torch.bfloat16

        # --- Step 1: Get the rotation matrix (freqs_cis) ---
        #    freqs_cis will be of type complex64.
        
        if seq_start + seq_len <= self.precompute_threshold:
            # freqs_cis_precomputed is already complex64.
            freqs_cis = self.freqs_cis_precomputed[:, :, seq_start:seq_start+seq_len, :]
        elif seq_start >= self.precompute_threshold:
            # Here, _compute_dynamic_freqs_cis should return complex64.
            freqs_cis = self._compute_dynamic_freqs_cis(seq_len, seq_start, x.device)
        else:
            # (Concatenation logic...)
            cached_part = self.freqs_cis_precomputed[:, :, seq_start:self.precompute_threshold, :]
            dynamic_len = seq_len - (self.precompute_threshold - seq_start)
            dynamic_part = self._compute_dynamic_freqs_cis(dynamic_len, self.precompute_threshold, x.device)
            freqs_cis = torch.cat([cached_part, dynamic_part], dim=2)

        # --- Step 2: Perform complex number operations in high precision (float32) ---
        #    This is to ensure numerical stability and compatibility.

        # 2a. Convert the input x to float32 and then to a complex number.
        x_float = x.to(torch.float32)
        x_complex = torch.view_as_complex(x_float.reshape(*x.shape[:-1], -1, 2))
        
        # 2b. Ensure the rotation matrix is also a float32 complex number (complex64)
        #    and select the correct half of the dimensions.
        freqs_cis_float = freqs_cis.to(device=x.device, dtype=torch.complex64)
        freqs_to_apply = freqs_cis_float[:, :, :, :self.dim//2]
        
        # --- Step 3: Perform the rotation (complex multiplication) ---
        x_rotated_complex = x_complex * freqs_to_apply
        
        # --- Step 4: Convert the result back to a real number ---
        x_rotated_float = torch.view_as_real(x_rotated_complex).flatten(3)

        # --- Step 5: Finally, convert the result back to the model's original low-precision type ---
        return x_rotated_float.to(dtype=original_dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, q_pos_start: int, k_pos_start: int) -> Tuple[torch.Tensor, torch.Tensor]:
        q_rotated = self._apply_rotary_emb(q, seq_start=q_pos_start)
        k_rotated = self._apply_rotary_emb(k, seq_start=k_pos_start)
        return q_rotated, k_rotated
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

"""
Utility functions for testing.
"""
import logging
import torch

# tests/utils_test.py
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(results, context_lengths, depth_percentiles, model_name):
    """
    Plots and saves a heatmap of the results.
    """
    # Flip the results matrix vertically for intuitive plotting
    scores = np.array([list(res.values()) for res in results.values()])
    scores = np.flipud(scores)

    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(scores, cmap='viridis', vmin=0, vmax=1)

    fig.colorbar(cax, label='Accuracy Score')

    ax.set_xlabel("Context Length (Number of Tokens)")
    ax.set_ylabel("Needle Depth in Document (%)")

    ax.set_xticks(np.arange(len(context_lengths)))
    ax.set_yticks(np.arange(len(depth_percentiles)))
    
    ax.set_xticklabels(context_lengths, rotation=45, ha="left")
    ax.set_yticklabels(reversed(depth_percentiles)) # Match flipped matrix

    # Add score annotations to each cell
    for i in range(len(depth_percentiles)):
        for j in range(len(context_lengths)):
            score = scores[i, j]
            color = "white" if score < 0.5 else "black"
            ax.text(j, i, f"{score:.2f}", ha="center", va="center", color=color)

    plt.title(f"Needle in a Haystack Results for: {model_name}")
    plt.tight_layout()
    
    save_path = f"needle_results_{model_name.replace('/', '_')}.png"
    plt.savefig(save_path)
    print(f"\nHeatmap saved to {save_path}")
    plt.show()


def log_peak_memory_usage(
    writer, device, test_type, config_path, cache_on, seq_len
):
    """
    Logs the peak GPU memory usage.
    """
    if torch.cuda.is_available():
        peak_mem_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_mem_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)

        # logging.info(
        #     "  Peak GPU Memory Allocated: %.4f GB, Reserved %.4f GB",
        #     peak_mem_alloc_gb, peak_mem_reserved_gb,
        # )

        writer.add_scalars(
            main_tag=(
                f"Performance/{test_type}/{config_path}/"
                f"gpu_peak_memory_gb/cache_on_{cache_on}"
            ),
            tag_scalar_dict={
                "Allocated": peak_mem_alloc_gb,
                "Reserved": peak_mem_reserved_gb,
            },
            global_step=seq_len,
        )

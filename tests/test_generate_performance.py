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
This script tests the generation performance of a model.
"""
import os
import time
import logging
import argparse
import torch
import numpy as np
from torch.amp import autocast

from tests.prepare_test import prepare_test
from tests.utils_test import log_peak_memory_usage


def do_generate(
    test_type,
    config_path,
    model,
    device,
    writer,
    config,
    initial_seq_len=10_000,
    num_new_tokens=4096,
    cache_on=True
):
    """
    Run the generation process and measure performance.
    """
    model.eval()

    vocab_size = config['model']['model_args']['vocab_size']
    input_ids = torch.randint(
        0, vocab_size - 10, size=(1, initial_seq_len),
        dtype=torch.long, device=device
    )
    input_tensor = torch.tensor(input_ids, device=device)
    
    logging.info("torch.cuda.empty_cache()")
    # Empty the cache to prevent fragmentation from slowing down allocation
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for _ in range(num_new_tokens):
            seq_len = input_tensor.shape[1]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            start_time = time.time()

            with autocast(device.type):
                outputs = model(
                    input_tensor, cache_on=cache_on
                )

                logits = outputs

            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(
                next_token_logits, dim=-1, keepdim=True
            )

            id_in_cpu = next_token_id.item()
            end_time = time.time()

            input_tensor = torch.cat([input_tensor, next_token_id], dim=1)

            time_spend = end_time - start_time

            cache_memory_usage = model.get_cache_memory_usage()

            logging.info(
                "test_type: %s %s, len: %d, cache_on: %s, spend time: %f, "
                "next_token_id: %d",
                test_type, config_path, seq_len, cache_on,
                time_spend, id_in_cpu
            )
            logging.info("cache_memory_usage: %s", cache_memory_usage)

            writer.add_scalar(
                f"Performance/{test_type}/{config_path}/cache_memory_usage/"
                f"cache_on_{cache_on}",
                cache_memory_usage['GB'],
                seq_len
            )

            log_peak_memory_usage(
                writer, device, test_type, config_path,
                cache_on, seq_len
            )

            writer.add_scalar(
                f"Performance/{test_type}/{config_path}/generate_time_cost/"
                f"cache_on_{cache_on}",
                time_spend,
                seq_len
            )

    model.clean_all_cache()


def main():
    """
    Main function to run the performance test.
    """
    
    parser = argparse.ArgumentParser(description="Test a Transformer model.")
    parser.add_argument('--config', type=str, help="YAML configuration file.")
    args = parser.parse_args()

    config_path = args.config
    cache_list = [True, False]
    num_new_tokens = 6
    
    model, device, config, writer = prepare_test(
        os.path.join("configs", os.path.basename(config_path))
    )

    for cache_on in cache_list:
        try:
            for initial_seq_len in range(1, 1_0000_0000, 1_0000):
                test_type = f"initial_seq_len_{initial_seq_len}"

                torch.cuda.synchronize()
                
                do_generate(
                    test_type, config_path, model, device, writer, config,
                    initial_seq_len, num_new_tokens, cache_on
                )
        except RuntimeError as e:
            logging.info("xxxxxxxxxxxxxxx test fail: %s xxxxxxxxxxxxxxx", e)


if __name__ == "__main__":
    main()

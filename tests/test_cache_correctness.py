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
import random

from tests.prepare_test import prepare_test
from tests.utils_test import log_peak_memory_usage


def do_cache_correctness_check(
    test_type,
    config_path,
    model,
    device,
    writer,
    config,
    input_tensor,
    num_new_tokens=4096,
    cache_on=True
):
    """
    Run the generation process and measure performance.
    """
    model.eval()
    
    with torch.no_grad():
        for _ in range(num_new_tokens):
            with autocast(device.type):
                outputs = model(
                    input_tensor, cache_on=cache_on
                )

                logits = outputs

            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(
                next_token_logits, dim=-1, keepdim=True
            )

            input_tensor = torch.cat([input_tensor, next_token_id], dim=1)

    model.clean_all_cache()
    
    return input_tensor

def main():
    """
    Main function to run the performance test.
    """
    
    parser = argparse.ArgumentParser(description="Test a Transformer model.")
    parser.add_argument('--config', type=str, help="YAML configuration file.")
    args = parser.parse_args()

    config_path = args.config
    initial_seq_len = random.randint(1000, 3000)
    num_new_tokens = random.randint(128, 512)
    
    model, device, config, writer = prepare_test(
        os.path.join("configs", os.path.basename(config_path))
    )

    vocab_size = config['model']['model_args']['vocab_size']
    input_ids = torch.randint(
        0, vocab_size - 10, size=(1, initial_seq_len),
        dtype=torch.long, device=device
    )
    
    input_tensor = torch.tensor(input_ids, device=device) 
    
    test_type = "cache_correctness"
    
    logging.info(f"{config_path} test cache on")
    out_ids_cache_on = do_cache_correctness_check(
        test_type, config_path, model, device, writer, config,
        input_tensor, num_new_tokens, cache_on=True
    )
    
    logging.info(f"{config_path} test cache off")
    out_ids_cache_off = do_cache_correctness_check(
        test_type, config_path, model, device, writer, config,
        input_tensor, num_new_tokens, cache_on=False
    )
    
    logging.info(f"{out_ids_cache_on.size()}")
    
    flag_equal = torch.equal(out_ids_cache_on, out_ids_cache_off)
    if not flag_equal:
        while True:
            logging.info(f"{config_path} cache correctness failed")
            time.sleep(3);

    logging.info(f"{config_path} cache correctness success")
    print(out_ids_cache_on)
    assert out_ids_cache_on.shape[1] == (initial_seq_len + num_new_tokens)
    
if __name__ == "__main__":
    main()

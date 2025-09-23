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
This script prepares the testing environment for a model.
"""
import os
import logging
import json
import yaml
import torch

# Refactored imports
from src.utils.utils import get_tensorboard_logger, setup_logging_logger
from src.utils.utils import create_model
from src.utils.checkpoint import load_checkpoint


def prepare_test(config_path, is_sft=False):
    """
    Prepares the testing environment for a model.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        tuple: A tuple containing the model, device, config, and writer.
    """
    torch.cuda.empty_cache()

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        logging.info("config: %s", config)

    log_dir = os.path.join("results", "tests", "logging_log")
    setup_logging_logger(log_dir, config['experiment_name'])

    # --- 1. Setup ---
    log_dir = os.path.join("results", "tests", "tensorboard_log")
    writer = get_tensorboard_logger(log_dir, config['experiment_name'])

    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model = create_model(config)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info("Model: %s", config['model']['class'])
    logging.info("Number of parameters: %.2fM", num_params)
    writer.add_text("model/parameters", f"{num_params:.2f}M")
    writer.add_text("config/all", f"<pre>{json.dumps(config, indent=2)}</pre>")

    model = load_checkpoint(model, device, config, is_sft)

    if config['hardware']['compile']:
        logging.info("Compiling the model with torch.compile()...")
        model = torch.compile(model)

    return model, device, config, writer

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
This script is for training a Transformer model.
It handles the entire training pipeline, including data loading,
model creation, training loop, evaluation, and checkpointing.
"""
import argparse
import json
import logging
import os
import time

import torch
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import DataLoader

# pylint: disable=import-error
from src.dataset.data_loader import get_data_loaders
from src.utils.checkpoint import load_train_checkpoint, save_checkpoint
from src.utils.utils import create_model, get_tensorboard_logger
from src.utils.utils import try_release_gpu_mem, logging_memory
from transformers import get_scheduler


# ==========================================================================================
# This function remains unchanged. It's our tool for measuring memory at a given batch size.
# ==========================================================================================
def profile_memory_for_batch_size(  # pylint: disable=too-many-arguments,too-many-locals
    model, optimizer, lr_scheduler, scaler, criterion, device, data_loader, config
):
    # ... (Your existing function code here, no changes needed) ...
    model.to(device)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    max_profile_steps = 32
    for i, batch in enumerate(data_loader):
        if i >= max_profile_steps:
            break
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        from torch.amp import autocast
        with autocast(device.type):
            outputs = model(x=inputs)
            loss = criterion(
                outputs.view(-int(outputs.size(-1))), labels.view(-1)
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config['training']['grad_clip']
        )
        scaler.step(optimizer)
        lr_scheduler.step()
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    return peak_memory_bytes

# ==========================================================================================
# NEW IMPLEMENTATION using Binary Search
# ==========================================================================================
def get_optimal_batch_size_binary_search(  # pylint: disable=too-many-arguments,too-many-locals
    model, optimizer, lr_scheduler, scaler, criterion, device, config, train_dataset
):
    """
    Finds the optimal batch size that fits into GPU memory using Binary Search.
    This is the most efficient search method for this monotonic problem.
    """
    if not torch.cuda.is_available():
        logging.info("CUDA is not available. Cannot perform GPU profiling.")
        return config['training']['per_device_train_batch_size']

    logging.info("Starting memory profiling for batch_size using Binary Search...")

    # --- 1. Define Search Boundaries and Target ---
    safety_margin = config['training']['gpu_memory_limit']
    total_memory_bytes = torch.cuda.get_device_properties(device).total_memory
    target_memory_bytes = total_memory_bytes * safety_margin
    
    low = 1
    high = config.get('training', {}).get('max_batch_size_search', 1024) # Set a reasonable upper bound
    optimal_bs = 0 # Start with 0, meaning no batch size has been found to fit yet

    logging.info(
        "Searching for optimal batch size in range [%d, %d] to fit under %.2f GB (%.2f GB total * %.2f margin)",
        low, high, target_memory_bytes / (1024**3), total_memory_bytes / (1024**3), safety_margin
    )

    # --- 2. Helper function to profile memory safely ---
    memo = {} # Memoization to cache results
    def _profile_bs(bs):
        if bs in memo:
            return memo[bs]
        if bs < 1:
             return float('inf')
        
        try:
            loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
            memory_bytes = profile_memory_for_batch_size(
                model, optimizer, lr_scheduler, scaler, criterion, device, loader, config
            )
            logging.info(f"  - Profiling bs={bs}: PASSED, Peak Memory: {memory_bytes / (1024**2):.2f} MB")
            memo[bs] = memory_bytes
            return memory_bytes
        except torch.cuda.OutOfMemoryError:
            logging.info(f"  - Profiling bs={bs}: FAILED (Out of Memory)")
            memo[bs] = float('inf')
            return float('inf')
        finally:
            del loader
            torch.cuda.empty_cache()

    # --- 3. Binary Search Loop ---
    while low <= high:
        mid = (low + high) // 2
        if mid == 0: # Avoid getting stuck if low is 0
            break
            
        mem_mid = _profile_bs(mid)
        
        if mem_mid <= target_memory_bytes:
            # `mid` is a viable batch size. Store it as our current best
            # and try to find a larger one by searching in the right half.
            optimal_bs = mid
            low = mid + 1
        else:
            # `mid` is too large and caused an OOM. The optimal size must
            # be smaller, so search in the left half.
            high = mid - 1
            
    if optimal_bs == 0:
        logging.error("Could not find any batch size that fits in the target memory, even bs=1. Check your model size and memory limit.")
        # Fallback to a safe value of 1, though it will likely fail.
        optimal_bs = 1

    logging.info("\n--- Binary Search Report ---")
    logging.info(f"Target memory: {target_memory_bytes / (1024**3):.2f} GB")
    logging.info(f"Found optimal batch size: {optimal_bs}")
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    return optimal_bs


def evaluate(model, eval_loader, device, writer, epoch):
    """
    Evaluates the model on the evaluation dataset.
    """
    model.eval()
    losses = []
    eval_progress_bar = tqdm(eval_loader, desc="Evaluating", leave=False)

    for batch in eval_progress_bar:
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(
                x=inputs, cache_on=False, evaluate_mode=True
            )
            full_logits = outputs

            loss = F.cross_entropy(
                full_logits.view(-1, full_logits.size(-1)), labels.view(-1)
            )
            losses.append(loss.item())

    eval_loss = torch.tensor(losses).mean()
    perplexity = torch.exp(eval_loss)

    writer.add_scalar('Loss/eval', eval_loss, epoch)
    writer.add_scalar('Metrics/perplexity', perplexity, epoch)
    logging.info(
        "Epoch %d: Eval Loss: %.4f, Perplexity: %.4f",
        epoch,
        eval_loss,
        perplexity,
    )

def train(config):  # pylint: disable=redefined-outer-name
    """
    Main training function.
    """
    # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    torch.cuda.empty_cache()

    # --- 1. Setup ---
    log_dir = os.path.join("results", "train")
    writer = get_tensorboard_logger(log_dir, config['experiment_name'])

    device = torch.device(
        config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    )
    logging.info("Using device: %s", device)

    # ---  Model ---
    # The factory function now creates your custom model
    # --- Data ---
    real_train_batch_size = None
    while True:
        model = create_model(config)
        model.to(device)

        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        logging.info("Model: %s", config['model']['class'])
        logging.info("Number of parameters: %.2fM", num_params)
        writer.add_text("model/parameters", f"{num_params:.2f}M")
        writer.add_text("config/all", f"<pre>{json.dumps(config, indent=2)}</pre>")
        writer.add_text("config/optimal_batch_size", f"{real_train_batch_size}")

        # --- 4. Optimizer and Scheduler ---
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(config['training']['beta1'], config['training']['beta2']),
        )
        criterion = F.cross_entropy

        train_dataset, _, train_loader, eval_loader, _ = get_data_loaders(
            config, real_train_batch_size
        )
        logging.info(
            "Vocabulary size: %d", config['model']['model_args']['vocab_size']
        )

        num_training_steps = (
            config['training']['num_train_epochs'] * len(train_loader)
        )
        lr_scheduler = get_scheduler(
            name=config['training']['lr_scheduler_type'],
            optimizer=optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=num_training_steps,
        )

        scaler = GradScaler(enabled=torch.cuda.is_available())

        model, optimizer, scaler, lr_scheduler, start_epoch, global_step = (
            load_train_checkpoint(
                model, optimizer, scaler, lr_scheduler, device, config
            )
        )

        if config['hardware']['compile']:
            logging.info("Compiling the model with torch.compile()...")
            model = torch.compile(model)

        # --- ----
        if real_train_batch_size is None:
            if config['training']['batch_method'] == "auto":
                real_train_batch_size = get_optimal_batch_size(
                    model,
                    optimizer,
                    lr_scheduler,
                    scaler,
                    criterion,
                    device,
                    config,
                    train_dataset,
                )
                if real_train_batch_size is None:
                    break
                logging.info(
                    "Need to adjust batch size, regenerate model and parameter..."
                )
            else:
                break
        else:
            break

    if real_train_batch_size is None:
        real_train_batch_size = config['training']['per_device_train_batch_size']

    virtual_train_batch_size = config['training']['virtual_train_batch_size']
    gradient_accumulation_steps = virtual_train_batch_size // real_train_batch_size
    if (virtual_train_batch_size % real_train_batch_size) != 0:
        raise ValueError("virtual_train_batch_size % real_train_batch_size != 0")

    # --- 5. Training Loop ---
    for epoch in range(start_epoch, config['training']['num_train_epochs'] + 1):
        torch.cuda.empty_cache()
        
        model.train()
        progress_bar = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch}/{config['training']['num_train_epochs']} "
                f"{config['experiment_name']}"
            ),
        )

        # reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        train_epoch_start_time = time.time()

        optimizer_step_flag = False

        def do_optimizer():
            # pylint: disable=cell-var-from-loop
            nonlocal global_step
            nonlocal optimizer_step_flag

            optimizer_step_flag = False
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['training']['grad_clip']
            )
            scaler.step(optimizer)
            lr_scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # Logging
            if global_step % config['training']['logging_steps'] == 0:
                writer.add_scalar(
                    'Loss/train', loss.item() * gradient_accumulation_steps, global_step
                )
                writer.add_scalar(
                    'LearningRate', lr_scheduler.get_last_lr()[0], global_step
                )
                progress_bar.set_postfix(
                    {"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"}
                )

        for i, batch in enumerate(progress_bar):
            try_release_gpu_mem(device)
            
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)

            with autocast(device.type):
                # The model forward now returns a dict
                outputs = model(x=inputs)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)), labels.view(-1)
                )
            # Normalize loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            optimizer_step_flag = True
            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                do_optimizer()

        if optimizer_step_flag:
            do_optimizer()

        train_epoch_end_time = time.time()
        writer.add_scalar(
            'time/train_per_epoch', train_epoch_end_time - train_epoch_start_time, epoch
        )

        # --- 6. Evaluation and Generation at Epoch End ---
        if epoch % config['training']['eval_steps'] == 0:
            evaluate(model, eval_loader, device, writer, epoch)
            # generate_and_log(model, tokenizer, device, writer, epoch, config)
            model.train()

        # --- 7. Save Checkpoint at Epoch End ---
        if epoch % config['training']['save_steps'] == 0:
            save_checkpoint(
                model, optimizer, scaler, lr_scheduler, epoch, global_step, config
            )

        logging_memory(device)

    writer.close()
    logging.info("Training finished.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    )

    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/fidelity_ultra_long_context.yaml',
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        logging.info("config: %s", config)

    train(config)

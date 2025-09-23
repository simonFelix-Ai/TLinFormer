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

import os
from torch.utils.tensorboard import SummaryWriter
import logging
import torch

def find_latest_checkpoint(config, is_sft=True):
    """查找最新的檢查點檔案。"""
    sub_dir = "sft" if is_sft else "train"
    checkpoint_dir = os.path.join("results", "checkpoints", sub_dir, config['experiment_name'])
    if not os.path.isdir(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
        
    # 按 epoch 數字排序找到最新的
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
    latest_ckpt_path = os.path.join(checkpoint_dir, checkpoints[0])
    logging.info(f"Found latest checkpoint: {latest_ckpt_path}")
    return latest_ckpt_path

def save_checkpoint(model, optimizer, scaler, lr_scheduler, epoch, global_step, config, is_sft=False):
    sub_dir = "sft" if is_sft else "train"
    checkpoint_dir = os.path.join("results", "checkpoints", sub_dir, config['experiment_name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    
    unwrapped_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(), # <--- 保存 GradScaler 狀態
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'config': config
    }, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")

def load_train_checkpoint(model, optimizer, scaler, lr_scheduler, device, config):
    """
    從最新的檢查點載入模型、優化器和訓練狀態。
    """
    start_epoch = 1
    global_step = 0
    
    latest_checkpoint_path = find_latest_checkpoint(config, False)    
    if latest_checkpoint_path:
        logging.info(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        
        # Hugging Face 的 scheduler.state_dict() 保存了其內部計數
        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info("Resumed LR scheduler state.")
        else:
            # 向後相容的 fallback
            logging.warning("Scheduler state not found in checkpoint. Manually stepping.")
            # 如果舊的檢查點沒有保存 scheduler 狀態，我們手動快進
            for _ in range(global_step):
                lr_scheduler.step()        
        
        # --- 恢復 GradScaler 狀態 ---
        # 檢查點中可能沒有 scaler 狀態（為了向後相容）
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logging.info("Resumed GradScaler state.")
        else:
            logging.warning("Scaler state not found in checkpoint, creating a new one.")
        
    else:
        logging.info("No checkpoint found. Starting training from scratch.")
        
    return model, optimizer, scaler, lr_scheduler, start_epoch, global_step

def load_sft_checkpoint(model, optimizer, scaler, lr_scheduler, device, config):
    """
    Loads model, optimizer, and training state from the latest checkpoint.
    Handles three distinct scenarios for SFT:
    1. Resuming an SFT run from an SFT checkpoint.
    2. Starting a new SFT run from a pre-training checkpoint (weights only).
    3. Starting a new SFT run from scratch.
    """
    start_epoch = 1
    global_step = 0
    
    # --- Step 1: Try to find a specific checkpoint for the current mode (SFT or Pre-train) ---
    latest_checkpoint_path = find_latest_checkpoint(config, True)
    
    # --- Step 2: Handle SFT logic specifically ---
    if  latest_checkpoint_path is None:
        # We are in SFT mode, but no SFT checkpoint was found.
        # Let's check for a pre-training checkpoint to initialize weights.
        logging.info("No SFT checkpoint found. Looking for a pre-training checkpoint to start SFT...")
        pretrain_checkpoint_path = find_latest_checkpoint(config, is_sft=False)
        
        if pretrain_checkpoint_path:
            logging.info(f"Initializing SFT from pre-training checkpoint: {pretrain_checkpoint_path}")
            checkpoint = torch.load(pretrain_checkpoint_path, map_location=device)
            
            # --- CRITICAL: Load ONLY the model weights ---
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            logging.info("Model weights loaded from pre-training checkpoint.")
            
            # --- DO NOT load optimizer, scheduler, epoch, etc. ---
            # We are starting a fresh SFT run, so we return the fresh objects.
            logging.info("Optimizer, LR scheduler, and counters are reset for the new SFT run.")
            return model, optimizer, scaler, lr_scheduler, start_epoch, global_step
        
        else:
            # No SFT checkpoint and no pre-training checkpoint found. Start SFT from scratch.
            logging.info("No pre-training checkpoint found either. Starting SFT from scratch.")
            return model, optimizer, scaler, lr_scheduler, start_epoch, global_step

    # --- Step 3: Handle the standard case (resuming a run from its own checkpoint) ---
    if latest_checkpoint_path:
        logging.info(f"Resuming SFT training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        
        # --- Load EVERYTHING to resume seamlessly ---
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 1) + 1  # Use .get for safety
        global_step = checkpoint.get('global_step', 0)
        
        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info("Resumed LR scheduler state.")
        else:
            logging.warning("Scheduler state not found. Manually stepping (less precise).")
            for _ in range(global_step):
                lr_scheduler.step()        
        
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logging.info("Resumed GradScaler state.")
        else:
            logging.warning("Scaler state not found, creating a new one.")

        logging.info(f"Resuming from Epoch {start_epoch}, Global Step {global_step}")
        return model, optimizer, scaler, lr_scheduler, start_epoch, global_step
        
    # --- Step 4: No checkpoint found at all for the current mode ---
    logging.info(f"No SFT checkpoint found. Starting SFT from scratch.")
    return model, optimizer, scaler, lr_scheduler, start_epoch, global_step

def load_checkpoint(model, device, config, is_sft=False):
    """
    從最新的檢查點載入模型、優化器和訓練狀態。
    """
    
    latest_checkpoint_path = find_latest_checkpoint(config, is_sft=is_sft)
    if latest_checkpoint_path is None and is_sft:
        latest_checkpoint_path = find_latest_checkpoint(config, is_sft=False)
    
    if latest_checkpoint_path:
        logging.info(f"Resuming model from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    else:
        logging.info("No checkpoint found.")

    return model
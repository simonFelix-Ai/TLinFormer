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
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import importlib
import sys
import hashlib
from transformers import AutoTokenizer

def get_tensorboard_logger(log_dir, name):
    """
    Initializes a TensorBoard SummaryWriter.
    """
    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique subdirectory for this run based on the experiment name
    run_dir = os.path.join(log_dir, name)
    
    print(f"TensorBoard log directory: {run_dir}")
    return SummaryWriter(log_dir=run_dir)

def setup_logging_logger(log_dir, name, level=logging.INFO, force=True):
    """
    配置根 logger，使其同时输出到文件和控制台。
    增加 'force' 参数来处理预先存在的 handler。
    """
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")
    
    logger = logging.getLogger()
    
    # 诊断信息
    print(f"--- [Logger Setup] Current handlers: {logger.handlers}")
    
    if logger.hasHandlers():
        if not force:
            print("--- [Logger Setup] Logger already configured. Skipping setup.")
            return
        else:
            print("--- [Logger Setup] 'force=True', removing existing handlers.")
            # 清空所有现有的 handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

    logger.setLevel(level)
    print(f"--- [Logger Setup] Setting logger level to {level}")
    print(f"--- [Logger Setup] Attempting to log to file: {os.path.abspath(log_file)}")

    # 创建 formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # 创建 file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    
    # 创建 stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    # 添加 handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logging.info("Root logger setup complete. Should log to console and file.")

def create_model(config):
    """
    Dynamically imports and creates a model based on the configuration.
    It looks for a factory function named 'create_<model_name>_model'
    in the corresponding module 'src.models.<model_name>'.
    """
    module_path = config['model']['module']
    class_name = config['model']['class']
    
    try:
        # Dynamically import the module
        logging.info(f"Creating model using factory: {class_name}")
        
        # 1. 动态导入模块
        module = importlib.import_module(module_path)
        
        # 2. 从模块中获取类对象
        target_class = getattr(module, class_name)
        
        # 3. 实例化类
        model = target_class(config)
        
    except (ImportError, AttributeError) as e:
        logging.error(f"Could not find or use model factory for '{module_path}'. ")
        raise e


    return model


def get_tensor_bytes_recursively(obj):
    """
    递归地遍历一个对象，计算其中所有 PyTorch 张量的总字节数。
    """
    total_bytes = 0
    
    if isinstance(obj, torch.Tensor):
        # Base Case: 如果对象是张量，计算其大小并返回
        return obj.nelement() * obj.element_size()
    
    elif isinstance(obj, dict):
        # Recursive Step: 如果是字典，遍历其值
        for value in obj.values():
            total_bytes += get_tensor_bytes_recursively(value)
            
    elif isinstance(obj, (list, tuple)):
        # Recursive Step: 如果是列表或元组，遍历其元素
        for item in obj:
            total_bytes += get_tensor_bytes_recursively(item)
            
    # 如果是其他类型 (int, str等)，则不增加字节数，直接返回当前的 total_bytes (即0)
    return total_bytes    

def load_tokenizer(config):
    data_config = config['data']
    
    logging.info(f"Loading tokenizer '{data_config['tokenizer_name']}' for SFT...")
    tokenizer = AutoTokenizer.from_pretrained(data_config['tokenizer_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # logging.info("Added a [PAD] token to the tokenizer.")
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def try_release_gpu_mem(device):
    if not torch.cuda.is_available():
        return
    
    total = torch.cuda.get_device_properties(device).total_memory
    rev = torch.cuda.max_memory_reserved(device)
    if rev > (total * 0.96):
        logging.info("Peak GPU Memory Reserved: %.4f GB", rev/(1024**3))
        logging.info("torch.cuda.empty_cache()")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
def build_prompt(instruction, context=""):
    """Builds the prompt in the exact format the model was trained on."""
    if context and context.strip():
        return f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:"
    
def logging_memory(device):
    if torch.cuda.is_available():
        peak_memory_allocated_gb = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
        )
        peak_memory_reserved_gb = (
            torch.cuda.max_memory_reserved(device) / (1024**3)
        )
        
        logging.info(
            "  Peak GPU Memory Allocated: %.4f GB", peak_memory_allocated_gb
        )
        logging.info(
            "  Peak GPU Memory Reserved:  %.4f GB", peak_memory_reserved_gb
        )

def get_quick_fingerprint(tensor):
    """
    Gets a simple, non-unique fingerprint of a tensor's values.
    """
    if not isinstance(tensor, torch.Tensor):
        return "N/A"
    # 计算均值、标准差和L1范数的和，作为一个粗略的指纹
    # .item() 将单元素张量转换为Python数字
    val_sum = tensor.sum().item()
    val_mean = tensor.mean().item()
    val_std = tensor.std().item()
    return f"sum={val_sum:.6f}_mean={val_mean:.6f}_std={val_std:.6f}"

def get_tensor_hash(tensor):
    """
    Computes a hash for a PyTorch tensor based on its values.
    """
    if not isinstance(tensor, torch.Tensor):
        return None
        
    # 1. 确保张量在CPU上，并转换为Numpy数组
    tensor_cpu = tensor.detach().cpu().numpy()
    
    # 2. 将数组设置为只读，这是一个好习惯
    tensor_cpu.flags.writeable = False
    
    # 3. 获取数组的字节表示
    tensor_bytes = tensor_cpu.tobytes()
    
    # 4. 使用SHA256计算哈希值
    hasher = hashlib.sha256(tensor_bytes)
    
    # 5. 返回十六进制的哈希摘要
    return hasher.hexdigest()
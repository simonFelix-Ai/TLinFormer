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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import os
import pickle

from src.utils.utils import build_prompt

# --- No changes needed for this class ---
class PackedDataset(Dataset):
    """
    A Dataset that takes a pre-tokenized stream of token IDs and repacks it
    into fixed-size chunks for training.
    """
    def __init__(self, token_stream, max_len):
        self.max_len = max_len
        self.samples = []
        self._repack(token_stream)

    def _repack(self, token_stream):
        """
        Repacks the long token stream into smaller, fixed-size samples.
        Each sample has a length of max_len + 1 to provide both input_ids and labels.
        """
        logging.info(f"Repacking the token stream into chunks of size {self.max_len}...")
        num_samples = (len(token_stream) - 1) // self.max_len
        
        for i in range(num_samples):
            start_idx = i * self.max_len
            end_idx = start_idx + self.max_len + 1
            chunk = token_stream[start_idx:end_idx]
            self.samples.append(chunk)

        logging.info(f"Created {len(self.samples):,} packed samples.")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a single sample, where 'labels' are the 'input_ids' shifted by one.
        """
        tokens = self.samples[idx]
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:]
        }

def _get_or_create_token_stream(data_config, tokenizer, split, packed_dataset_cache_dir, use_cache):
    """
    Handles Tier 1 Caching: Tokenizes a dataset split and caches the resulting token stream.
    This version is optimized for large RAM systems by processing the entire dataset in memory.
    """
    # Define a subdirectory for token stream caches within the main cache directory
    token_stream_cache_dir = packed_dataset_cache_dir
    if use_cache:
        os.makedirs(token_stream_cache_dir, exist_ok=True)
    
    # Generate a unique cache path for the token stream
    dataset_path_str = data_config['dataset_path'].replace('/','_')
    dataset_name_str = data_config['dataset_name']
    if dataset_name_str is None:
        dataset_name_str = ""
    else:
        dataset_name_str = dataset_name_str.replace('/','_')
    tokenizer_name = tokenizer.name_or_path.replace('/', '_')
    cache_filename = f"token_stream_{dataset_path_str}_{dataset_name_str}_{tokenizer_name}_{split}.pt"
    cache_path = os.path.join(token_stream_cache_dir, cache_filename)

    # --- Tier 1 Caching: Load from cache if it exists ---
    if use_cache and os.path.exists(cache_path):
        logging.info(f"Loading tokenized '{split}' stream from Tier 1 cache: {cache_path}")
        try:
            token_stream = torch.load(cache_path)
            logging.info(f"Total tokens in '{split}' stream: {len(token_stream):,}")
            return token_stream
        except Exception as e:
            logging.warning(f"Could not load token stream cache file '{cache_path}' due to an error: {e}. Rebuilding...")

    logging.info(f"Token stream cache for '{split}' not found or invalid. Creating from scratch...")
    
    try:
        dataset_path = data_config['dataset_path']
        dataset_name = data_config['dataset_name']
        logging.info(f"Attempting to load dataset {dataset_path} {dataset_name} from local cache (offline mode)...")
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "10"
        raw_dataset = load_dataset(
            dataset_path,
            dataset_name,
            split=split,
            cache_dir=data_config.get('dataset_cache_dir'),
            download_mode="reuse_cache_if_exists"
        )
        logging.info(f"Loaded dataset: {dataset_path} {dataset_name}, {raw_dataset.num_rows} rows for split '{split}'")
    except Exception as e:
        logging.warning(f"Failed to load dataset: {e}.")
    
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'
        })
        logging.info("Tokenizer eos_token not found. Added '<|endoftext|>'.")
        
    # --- START OF OPTIMIZATION ---                                       
    if not tokenizer.is_fast:
        logging.warning("Using a slow tokenizer. For massive speed-ups, use a fast tokenizer.")    

    if tokenizer.eos_token_id is None:
        # 如果没有，通常会添加一个。对于 GPT2，它已经有了。
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        logging.info("Tokenizer eos_token not found. Added '<|endoftext|>'.")
        
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)
    
    # This is the single, efficient call.
    # It processes the entire dataset without loading it all into RAM.
    # It streams small batches from disk to the parallel workers.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,           # Process data in batches (default size 1000)
        batch_size=(os.cpu_count() * 10),
        num_proc=1,      # tokenizer parallel is True, so just set it to 1
        remove_columns=raw_dataset.column_names, # Remove old columns to save space on disk
        desc=f"Tokenizing split '{split}'" # Provides a nice TQDM progress bar
    )
    
    # At this point, `tokenized_dataset` contains one column 'all_tokens' where each row
    # is a list of tokens from one original batch. We need to combine these rows.
    logging.info("Flattening results from all processed batches...")
    all_tokens = []
    eos_token_id = tokenizer.eos_token_id
    
    for token_list in tqdm(tokenized_dataset['input_ids'], desc=f"Flattening tokens for split '{split}'"):
        all_tokens.extend(token_list)
        if eos_token_id is not None:
            all_tokens.append(eos_token_id)

    token_stream = torch.tensor(all_tokens, dtype=torch.long)
    logging.info(f"Total tokens in '{split}' stream: {len(token_stream):,}")

    # --- Save to Tier 1 Cache ---
    if use_cache:
        logging.info(f"Saving tokenized '{split}' stream to Tier 1 cache: {cache_path}")
        torch.save(token_stream, cache_path)

    return token_stream

def _get_packed_cache_paths(data_config, max_len, validation_ratio, packed_dataset_cache_dir):
    """ Generates consistent cache paths for packed datasets based on configuration parameters. """
    dataset_path_str = data_config['dataset_path'].replace('/','_')
    dataset_name_str = data_config['dataset_name']
    if dataset_name_str is None:
        dataset_name_str = ""
    else:
        dataset_name_str = dataset_name_str.replace('/','_')
    tokenizer_name = data_config['tokenizer_name'] 
    
    config_str = f"{dataset_path_str}-{dataset_name_str}-{tokenizer_name}-{max_len}-{validation_ratio}"
        
    train_path = os.path.join(packed_dataset_cache_dir, f"packed_{config_str}_train.pt")
    eval_path = os.path.join(packed_dataset_cache_dir, f"packed_{config_str}_eval.pt")
    
    return train_path, eval_path

def _get_packed_dataset(data_config, max_len, validation_ratio, packed_dataset_cache_dir, use_cache):
    """ Attempts to load the final packed train and eval datasets from the Tier 2 cache. """
    if not use_cache:
        return None, None
    
    train_path, eval_path = _get_packed_cache_paths(data_config, max_len, validation_ratio, packed_dataset_cache_dir)

    if os.path.exists(train_path) and os.path.exists(eval_path):
        logging.info(f"Loading packed datasets from Tier 2 cache...")
        try:
            train_dataset = torch.load(train_path, weights_only=False)
            eval_dataset = torch.load(eval_path, weights_only=False)
            logging.info("Successfully loaded packed datasets from cache.")
            return train_dataset, eval_dataset
        except (ModuleNotFoundError, pickle.UnpicklingError) as e:
            logging.warning(f"Could not load cache file due to an error: {e}. Rebuilding dataset.")
            return None, None

    logging.info("Packed dataset cache (Tier 2) not found.")
    return None, None

def _create_packed_dataset(data_config, tokenizer, cache_dir, use_cache):
    """ Creates, packs, and caches the train and eval datasets. """
        
    # 3. Prepare token streams
    train_token_stream = _get_or_create_token_stream(
        data_config, tokenizer, 'train', cache_dir, use_cache
    )
    
    all_splits = get_dataset_split_names(
        data_config["dataset_path"],
        data_config["dataset_name"],
        cache_dir=data_config.get('dataset_cache_dir')
    )
    has_validation_split = 'validation' in all_splits

    if has_validation_split:
        val_token_stream = _get_or_create_token_stream(
            data_config, tokenizer, 'validation', cache_dir, use_cache
        )
    else:
        logging.info(f"No validation split found. Splitting train set with ratio {data_config['validation_ratio']}.")
        validation_ratio = data_config['validation_ratio']
        if not (0 < validation_ratio < 1):
            raise ValueError("validation_ratio must be between 0 and 1.")
        val_size = int(len(train_token_stream) * validation_ratio)
        val_token_stream = train_token_stream[-val_size:]
        train_token_stream = train_token_stream[:-val_size]

    # 4. Create packed datasets
    max_len = data_config['max_train_seq_len']
    train_dataset = PackedDataset(train_token_stream, max_len)
    eval_dataset = PackedDataset(val_token_stream, max_len)

    if use_cache:
        train_path, eval_path = _get_packed_cache_paths(
            data_config, max_len, data_config['validation_ratio'], cache_dir
        )
        logging.info(f"Saving final packed datasets to Tier 2 cache...")
        torch.save(train_dataset, train_path)
        torch.save(eval_dataset, eval_path)

    return train_dataset, eval_dataset

# --- The Definitive Data Loading Function ---
def get_data_loaders(config, optimal_bs=None):
    data_config = config['data']
    cache_config = data_config.get('packed_dataset_cache', {})
    use_cache = cache_config.get('use_cache', True)
    packed_dataset_cache_dir = cache_config.get('dir', './dataset_cache/packed_dataset_cache')
    if use_cache:
        os.makedirs(packed_dataset_cache_dir, exist_ok=True)

    # 1. Setup Tokenizer
    logging.info(f"Loading tokenizer '{data_config['tokenizer_name']}'...")
    tokenizer = AutoTokenizer.from_pretrained(data_config['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logging.info("Added a [PAD] token to the tokenizer.")
    
    # 2. Attempt to load packed datasets from cache, or create them
    max_len = data_config['max_train_seq_len']
    
    train_dataset, eval_dataset = _get_packed_dataset(
        data_config,
        max_len, 
        data_config['validation_ratio'], 
        packed_dataset_cache_dir, 
        use_cache
    )
    
    if not train_dataset or not eval_dataset:
       train_dataset, eval_dataset = _create_packed_dataset(data_config, tokenizer, packed_dataset_cache_dir, use_cache)

    if (config['training']['batch_method'] == "auto") and (optimal_bs is not None):
        real_train_bs, real_val_bs = optimal_bs, optimal_bs
    else:
        real_train_bs = config['training']['per_device_train_batch_size']
        real_val_bs = config['training']['per_device_eval_batch_size']
    logging.info(f"use batch size: train({real_train_bs}) eval({real_val_bs})")
    
    # 3. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=real_train_bs,
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=real_val_bs,
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )

    logging.info("tokenizer.vocab_size %s", tokenizer.vocab_size)
    
    return train_dataset, eval_dataset, train_loader, eval_loader, tokenizer


# --- The Definitive Data Loading Function ---
def get_token_streams(config):
    data_config = config['data']
    cache_config = data_config.get('packed_dataset_cache', {})
    use_cache = cache_config.get('use_cache', True)
    cache_dir = cache_config.get('dir', './dataset_cache/packed_dataset_cache')

    # 1. Setup Tokenizer
    logging.info(f"Loading tokenizer '{data_config['tokenizer_name']}'...")
    tokenizer = AutoTokenizer.from_pretrained(data_config['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # logging.info("Added a [PAD] token to the tokenizer.")

    train_token_stream = _get_or_create_token_stream(
        data_config, tokenizer, 'train', cache_dir, use_cache
    )
    
    val_token_stream = _get_or_create_token_stream(
        data_config, tokenizer, 'validation', cache_dir, use_cache
    )

    logging.info("tokenizer.vocab_size %s", tokenizer.vocab_size)
    
    return train_token_stream, val_token_stream, tokenizer


# --- Example Usage (Unchanged) ---
if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    base_config = {
        'data': {
            'dataset_path': 'brando/small-c4-dataset',
            'dataset_name': 'null',
            'dataset_cache_dir': './dataset_cache',
            'tokenizer_name': 'gpt2',
            'num_workers': 4,
            'packed_dataset_cache': { 'use_cache': True, 'dir': './dataset_cache/packed_dataset_cache' }
        },
        'training': { 'per_device_train_batch_size': 8, 'per_device_eval_batch_size': 8 },
        'model': { 'model_args': {} }
    }

    # Helper to create configs cleanly
    def create_config(base, max_len, val_ratio):
        new_config = dict(base)
        new_config['data'] = dict(base['data'])
        new_config['data']['max_train_seq_len'] = max_len
        new_config['data']['validation_ratio'] = val_ratio
        return new_config

    # --- Run 1: len=1024, val_ratio=0.03 ---
    print("\n" + "="*20 + " RUN 1: len=1024, val_ratio=0.03 " + "="*20)
    config_1 = create_config(base_config, max_len=1024, val_ratio=0.03)
    train_loader_1, eval_loader_1, _ = get_data_loaders(config_1)
    print(f"Run 1 -> Train batches: {len(train_loader_1)}, Eval batches: {len(eval_loader_1)}")

    # --- Run 2: len=512, val_ratio=0.03 ---
    print("\n" + "="*20 + " RUN 2: len=512, val_ratio=0.03 " + "="*20)
    config_2 = create_config(base_config, max_len=512, val_ratio=0.03)
    train_loader_2, eval_loader_2, _ = get_data_loaders(config_2)
    print(f"Run 2 -> Train batches: {len(train_loader_2)}, Eval batches: {len(eval_loader_2)}")

    # --- Run 3: len=1024, val_ratio=0.05 ---
    print("\n" + "="*20 + " RUN 3: len=1024, val_ratio=0.05 " + "="*20)
    config_3 = create_config(base_config, max_len=1024, val_ratio=0.05)
    train_loader_3, eval_loader_3, _ = get_data_loaders(config_3)
    print(f"Run 3 -> Train batches: {len(train_loader_3)}, Eval batches: {len(eval_loader_3)}")
    
    # --- Run 4: len=1024, val_ratio=0.03 (Again) ---
    print("\n" + "="*20 + " RUN 4: len=1024, val_ratio=0.03 (Again) " + "="*20)
    config_4 = create_config(base_config, max_len=1024, val_ratio=0.03)
    train_loader_4, eval_loader_4, _ = get_data_loaders(config_4)
    print(f"Run 4 -> Train batches: {len(train_loader_4)}, Eval batches: {len(eval_loader_4)}")

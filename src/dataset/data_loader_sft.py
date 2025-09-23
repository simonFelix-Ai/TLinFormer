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
import random
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
import logging
import os
from functools import partial

# Assume needles.py is in the same directory or accessible via path
from src.utils.utils import build_prompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SyntheticNeedleDataset(Dataset):
    """
    Generates synthetic "Needle in a Haystack" samples directly from a token stream.
    """
    def __init__(self, token_stream, tokenizer, num_samples, sft_config):
        self.token_stream = token_stream
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.sft_config = sft_config
        self.eos_token_id = self.tokenizer.eos_token_id
        self.ignore_index = -100
        
        # Pre-tokenize all needles to avoid tokenizing in the loop
        self._generate_needle_templates()
        self._pre_tokenize_needles()
        self._pre_generate_items()
        
    def _generate_needle_templates(self, num_templates=300):
        # Generate a large number of diverse templates
        self.needle_templates = []
        for _ in range(num_templates):
            num_digits = random.choice([1, 2, 3, 4, 5, 6, 7])
            if num_digits == 1:
                key_number = random.randint(0, 9)
            else:
                key_number = random.randint(10**(num_digits - 1), 10**num_digits - 1)
            
            key_str = str(key_number)
            
            # You can even randomize the question/needle text slightly
            key_synonym = random.choice(["key", "code", "passcode", "ID", "number", "sequence"])
            adjective = random.choice(["important", "secret", "critical", "access", "special"])
            
            template = {
                "needle_text": f"The {adjective} {key_synonym} is {key_str}.",
                "question_text": f"what is the {adjective} {key_synonym}?",
                "answer_text": key_str
            }
            self.needle_templates.append(template)
    
    def _generate_one_item(self):
        needle_data = random.choice(self.tokenized_needles)
        needle_tokens = needle_data['needle_tokens']
        
        # --- 2. Create the haystack from the token stream ---
        # Define a random length for the haystack to add variety
        haystack_len = random.randint(self.sft_config['synthetic_minlen'], self.sft_config['synthetic_maxlen'])
        
        # Take a random slice from the token stream
        start_idx = random.randint(0, len(self.token_stream) - haystack_len)
        haystack_tokens = self.token_stream[start_idx : start_idx + haystack_len].tolist()
        
        # Remove any EOS tokens from the middle of the haystack
        if self.eos_token_id is not None:
            haystack_tokens = [token for token in haystack_tokens if token != self.eos_token_id]

        # --- 3. Insert the needle tokens randomly into the haystack tokens ---
        insertion_point = random.randint(0, len(haystack_tokens))
        haystack_with_needle_tokens = haystack_tokens[:insertion_point] + needle_tokens + haystack_tokens[insertion_point:]

        # --- 4. Format the prompt and response, then tokenize them ---
        # Decode the haystack part to create the context string for the prompt
        # haystack_with_needle_text = self.tokenizer.decode(haystack_with_needle_tokens)

        prompt_text = build_prompt(needle_data['question_text'])
        response_text = needle_data['answer_text'] + self.tokenizer.eos_token

        tokenized_prompt = self.tokenizer(prompt_text, add_special_tokens=False)
        tokenized_response = self.tokenizer(response_text, add_special_tokens=False)
        
        # --- 6. Combine to create final input_ids and labels ---
        input_ids = tokenized_prompt['input_ids'] + tokenized_response['input_ids']
        prompt_len = len(tokenized_prompt['input_ids'])
        labels = ([self.ignore_index] * prompt_len) + tokenized_response['input_ids']

        # --- Prepare the final output, including the raw text for inspection ---
        output = {
            "haystack_with_needle": haystack_with_needle_tokens,
            "input_ids": input_ids[:-1],
            "labels": labels[1:],
            # We return the text for debugging/inspection if needed
            "prompt_text": prompt_text,
            "response_text": response_text.strip()
        }        

        return output
        
    def _pre_generate_items(self):
        self.all_items = []
        
        for _ in range(self.num_samples):
            one_item = self._generate_one_item()
            self.all_items.append(one_item)
            
    def _pre_tokenize_needles(self):
        self.tokenized_needles = []
        for item in self.needle_templates:
            # We add special formatting and tokenize. This is done only once.
            needle_str = f"\n*** {item['needle_text']} ***\n"
            self.tokenized_needles.append({
                "needle_tokens": self.tokenizer.encode(needle_str, add_special_tokens=False),
                "question_text": item['question_text'],
                "answer_text": item['answer_text']
            })
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # output = self.all_items[idx]
        output = self._generate_one_item()
        
        return output

def custom_needle_collate_fn(batch, pad_token_id):
    """
    A custom collate function that handles the specific output of SyntheticNeedleDataset.
    It pads 'haystack_with_needle' and 'prompt_with_response' to the max length in the batch.

    Args:
        batch (list): A list of dictionaries, where each dict is an output of __getitem__.
        pad_token_id (int): The token ID to use for padding.

    Returns:
        dict: A dictionary containing batched and padded tensors.
    """
    # --- 1. Find the max length for each key in the current batch ---
    max_haystack_len = 0
    max_labels_len = 0
    for sample in batch:
        if len(sample['haystack_with_needle']) > max_haystack_len:
            max_haystack_len = len(sample['haystack_with_needle'])
        if len(sample['labels']) > max_labels_len:
            max_labels_len = len(sample['labels'])            
            
    # --- 2. Initialize empty tensors (the "canvas") for padding ---
    batch_size = len(batch)
    padded_haystacks = torch.full((batch_size, max_haystack_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((batch_size, max_labels_len), pad_token_id, dtype=torch.long)
    padded_input_ids = torch.full((batch_size, max_labels_len), pad_token_id, dtype=torch.long)
    
    # Also collect the text fields
    prompt_texts = []
    response_texts = []

    # --- 3. Fill the tensors with data from the batch ---
    for i, sample in enumerate(batch):
        haystack_seq = sample['haystack_with_needle']
        labels_seq = sample['labels']
        input_ids_seq = sample['input_ids']
        
        # Copy data to the padded tensors
        padded_haystacks[i, :len(haystack_seq)] = torch.tensor(haystack_seq, dtype=torch.long)
        padded_labels[i, :len(labels_seq)] = torch.tensor(labels_seq, dtype=torch.long)
        padded_input_ids[i, :len(input_ids_seq)] = torch.tensor(input_ids_seq, dtype=torch.long)
                
        # Collect text
        prompt_texts.append(sample['prompt_text'])
        response_texts.append(sample['response_text'])

    # --- 4. Return the final batch dictionary with TENSORS ---
    return {
        "haystack_with_needle": padded_haystacks,
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "prompt_text": prompt_texts,       # These will be lists of strings
        "response_text": response_texts  # These will be lists of strings
    }

def get_sft_data_loaders_from_streams(config, train_token_stream, val_token_stream, tokenizer):
    """
    Creates SFT DataLoaders using the SyntheticNeedleDataset.
    """
    sft_config = config['sft']
        
    # Create the Synthetic Dataset instances
    num_train_samples = sft_config.get('num_synthetic_samples', 10000)
    train_dataset = SyntheticNeedleDataset(train_token_stream, tokenizer, num_train_samples, sft_config)
    
    num_eval_samples = int(num_train_samples * 0.05)
    eval_dataset = SyntheticNeedleDataset(val_token_stream, tokenizer, num_eval_samples, sft_config)

    collate_fn_with_padding = partial(custom_needle_collate_fn, pad_token_id=tokenizer.pad_token_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=sft_config['per_device_train_batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_with_padding
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=sft_config['per_device_eval_batch_size'],
        num_workers=0,
        collate_fn=collate_fn_with_padding
    )

    return train_loader, eval_loader
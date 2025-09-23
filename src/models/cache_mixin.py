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

# pylint: disable=too-many-positional-arguments
"""
This file
"""
import logging
import torch

# pylint: disable=import-error
from src.utils.utils import get_tensor_bytes_recursively

class CacheMixin:
    """Mixin for cache operations."""
    @staticmethod
    def create_share_status(config):
        """Creates the initial shared status dictionary."""
        return {
            "cache": {}, # the real kv and attention cache
            "cache_on": False,
            "config": config,
            "prev_context_len": 0,
            "context_changed": True
        }

    def _init_cache_state(self, share_status):
        """Links the module to a shared cache state."""
        self.unique_id = id(self)
        share_status['cache'][self.unique_id] = {}
        self.share_status = share_status

    def get_common_cache(self, key):
        """Gets past cache."""
        return self.share_status[key]
    
    def set_common_cache(self, key, val):
        """Gets past cache."""
        self.share_status[key] = val

    def is_cache_on(self):
        return self.share_status['cache_on']

    def set_cache_on(self, cache_on):
        self.share_status['cache_on'] = cache_on

    def get_unique_cache(self, key):
        """Gets past cache."""
        return self.share_status['cache'][self.unique_id].get(key, None)

    def set_unique_cache(self, key, val):
        """Sets current cache."""
        if self.is_cache_on():
            # logging.info(" set_unique_cache %s", key)
            self.share_status['cache'][self.unique_id][key] = val

    def is_cache_valid(self, key):
        """Checks if past cache is valid."""
        valid = key in self.share_status['cache'][self.unique_id]
        # logging.info("%s valid %s", key, valid)
        return valid

    def clean_all_cache(self):
        """Cleans all cache."""
        if not self.training:
            logging.info("clean all cache")

        for unique_id in self.share_status['cache'].keys():
            self.share_status['cache'][unique_id] = {}

    def clean_invalid_cache(self):
        """Cleans invalid cache."""
        # keep_cache_keys = ["context_origin_qkv"]
        # if self.is_cache_on():
        #     if not self.training:
        #         logging.info("clean invalid cache")

        # del_list = []
        # for unique_id in self.share_status['cache'].keys():
        #     for sub_key in self.share_status['cache'][unique_id]:
        #         if sub_key not in keep_cache_keys:
        #             del_list.append((unique_id, sub_key))

        # for unique_id, sub_key in del_list:
        #     del self.share_status['cache'][unique_id][sub_key]

        # only the bottom context_origin_qkv can be reused
        # for simplify, just remove all cache
        self.clean_all_cache()

    def dump_all_cache_keys(self, info):
        """Dumps all cache keys."""
        logging.info("%s %s start %s", '=' * 16, info, '=' * 16)

        for unique_id in self.share_status['cache'].keys():
            for sub_key in self.share_status['cache'][unique_id]:
                val = self.share_status['cache'][unique_id][sub_key]
                if "qkv" in sub_key:
                    (q, k, v) = val
                    logging.info("unique_id: %s, key: %s, val id: %s q,k,v size: (%s,%s,%s)",
                                unique_id, sub_key, id(val), q.size(),k.size(),v.size())                    
                else:
                    if isinstance(val, torch.Tensor):
                        val_size = val.size()
                        logging.info("unique_id: %s, key: %s, val id: %s size: %s",
                                    unique_id, sub_key, id(val), val_size)

        logging.info("%s %s end %s", '=' * 16, info, '=' * 16)

    def get_cache_memory_usage(self):       
        total_bytes = get_tensor_bytes_recursively(self.share_status['cache'])

        kb = total_bytes / 1024
        mb = kb / 1024
        gb = mb / 1024

        return {
            "bytes": total_bytes,
            "KB": round(kb, 3),
            "MB": round(mb, 3),
            "GB": round(gb, 3)
        }
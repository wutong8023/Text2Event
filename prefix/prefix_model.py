#!/usr/bin/env python
#
# Copyright the CoLL team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Intro: 
# Author: Tongtong Wu
# Time: Oct 14, 2021
"""
import torch
import inspect
import torch.nn as nn

from typing import NamedTuple
from collections import namedtuple
from argparse import Namespace
from functools import partial
from transformers import PreTrainedModel, PretrainedConfig, T5Config, GPT2Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from prefix.T5forPrefixGeneration import T5ForPrefixGeneration


def signature(f):
    r"""Get the function f 's input arguments. A useful gadget
    when some function slot might be instantiated into multiple functions.

    Args:
        f (:obj:`function`) : the function to get the input arguments.

    Returns:
        namedtuple : of args, default, varargs, keywords, respectively.s

    """
    sig = inspect.signature(f)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    keywords = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    keywords = keywords[0] if keywords else None
    defaults = [
                   p.default for p in sig.parameters.values()
                   if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                      and p.default is not p.empty
               ] or None
    argspec = namedtuple('Signature', ['args', 'defaults', 'varargs', 'keywords'])
    return argspec(args, defaults, varargs, keywords)


class PrefixEncoderDecoder(nn.Module):
    def __init__(self, model: PreTrainedModel, training_args: Namespace, num_token: int = 5,
                 prefix_dropout: float = 0.5):
        super(PrefixEncoderDecoder, self).__init__()
        self.model = model
        self.config = model.config
        self.training_args = training_args
        self.device = training_args.device
        self.tuning_type = training_args.tuning_type
        self.num_token = num_token
        
        # PLM-related parameters
        if isinstance(self.config, T5Config):
            self.n_layer = self.config.num_layers
            self.n_embd = self.config.d_model
            self.n_head = self.config.num_heads
            self.n_decoder_layer = self.config.num_decoder_layers
            self.match_n_decoder_layer = self.n_decoder_layer
        
        self.mid_dim = self.n_embd
        self.match_n_layer = self.n_layer
        self.match_n_head = self.n_head
        self.match_n_embd = self.n_embd // self.n_head
        self.prefix_dropout = prefix_dropout
        self.dropout = nn.Dropout(self.prefix_dropout)
        
        self.generate_parameters()  # prepare for parameter generation
        self.is_modified_model = False
        self.modify_model()
    
    def forward(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        prefix_past_key_values = self.get_past_key_values(batch_size)
        kwargs["prefix_key_values"] = prefix_past_key_values
        
        outputs = self.model(**kwargs)
        return outputs
    
    def generate_parameters(self):
        """
        get prompt encoder
        Returns:

        """
        self.input_tokens = nn.Parameter(torch.arange(self.num_token).long(),
                                         requires_grad=False).to(self.device)
        self.wte = nn.Embedding(self.num_token, self.n_embd).to(self.device)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_layer * 2 * self.n_embd)).to(self.device)
        
        self.decoder_input_tokens = nn.Parameter(torch.arange(self.num_token).long(),
                                                 requires_grad=False).to(self.device)
        self.decoder_wte = nn.Embedding(self.num_token, self.n_embd).to(self.device)
        self.decoder_control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_decoder_layer * 2 * self.n_embd)).to(self.device)
        
        pass
    
    def get_past_key_values(self, batch_size: int):
        # encoder
        input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        _, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(batch_size, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        
        # decoder
        decoder_input_tokens = self.decoder_input_tokens.unsqueeze(0).expand(batch_size, -1)
        decoder_temp_control = self.decoder_wte(decoder_input_tokens)
        decoder_past_key_values = self.decoder_control_trans(decoder_temp_control)  # bsz, seqlen, layer*emb
        _, decoder_seqlen, _ = decoder_past_key_values.shape
        decoder_past_key_values = decoder_past_key_values.view(batch_size, decoder_seqlen,
                                                               self.match_n_decoder_layer * 2, self.match_n_head,
                                                               self.match_n_embd)
        decoder_past_key_values = self.dropout(decoder_past_key_values)
        decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        
        return past_key_values, decoder_past_key_values
    
    def modify_model(self):
        if self.is_modified_model:
            return None
        
        if self.tuning_type == "prefix":
            for pp in self.model.parameters():
                pp.requires_grad = False
        
        if isinstance(self.model, T5ForPrefixGeneration):
            backup_encoder_forward_functions = []
            for i, layer_module in enumerate(self.model.encoder.block):
                backup_encoder_forward_functions.append(layer_module.layer[0].forward)
                
                def modified_encoder_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    if kwargs['past_key_value'] is None:
                        kwargs['past_key_value'] = kwargs["prefix_key_values"][0][layer_id]
                    
                    if kwargs['attention_mask'] is not None:
                        am = kwargs['attention_mask']
                        kwargs['attention_mask'] = torch.cat(
                            [torch.ones((*am.shape[:-1], self.num_token), dtype=am.dtype, device=am.device), am],
                            dim=-1)
                    return backup_encoder_forward_functions[layer_id](*args, **kwargs)
                
                layer_module.layer[0].forward = partial(modified_encoder_forward, layer_id=i)
            
            backup_decoder_self_attn_forward_functions = []
            backup_decoder_cross_attn_forward_functions = []
            for i, layer_module in enumerate(self.model.decoder.block):
                backup_decoder_self_attn_forward_functions.append(layer_module.layer[0].forward)
                
                def modified_decoder_self_attn_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    if kwargs['past_key_value'] is None:
                        kwargs['past_key_value'] = kwargs["prefix_key_values"][1][layer_id]
                    if kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1):
                        pass
                    elif kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(
                            -1) + self.num_token:
                        am = kwargs['attention_mask']
                        kwargs['attention_mask'] = torch.cat(
                            [torch.ones((*am.shape[:-1], self.num_token), dtype=am.dtype, device=am.device), am],
                            dim=-1)
                    else:
                        raise RuntimeError("Size not match: past length: {}, inputlength:{},\
                             attention mask length {}".format(kwargs['past_key_value'][0].size(-2),
                                                              args[0].size(-2), kwargs['attention_mask'].size(-1)))
                    return backup_decoder_self_attn_forward_functions[layer_id](*args, **kwargs)
                
                layer_module.layer[0].forward = partial(modified_decoder_self_attn_forward, layer_id=i)
                backup_decoder_cross_attn_forward_functions.append(layer_module.layer[1].forward)
                
                def modified_decoder_cross_attn_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    return backup_decoder_cross_attn_forward_functions[layer_id](*args, **kwargs)
                
                layer_module.layer[1].forward = partial(modified_decoder_cross_attn_forward, layer_id=i)
            
            self.backup_encoder_forward_functions = backup_encoder_forward_functions
            self.backup_decoder_self_attn_forward_functions = backup_decoder_self_attn_forward_functions
            self.backup_decoder_cross_attn_forward_functions = backup_decoder_cross_attn_forward_functions
    
    def generate(self, *args, **kwargs):
        batch_size = kwargs["attention_mask"].size(0)
        prefix_past_key_values = self.get_past_key_values(batch_size)
        kwargs["prefix_key_values"] = prefix_past_key_values
        return self.model.generate(*args, **kwargs)



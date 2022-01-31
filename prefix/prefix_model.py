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
import os
import json
import copy
import torch.nn as nn

from typing import NamedTuple
from collections import namedtuple
from argparse import Namespace
from functools import partial
from transformers import PreTrainedModel, PretrainedConfig, T5Config, GPT2Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.modeling_utils import *
from transformers.trainer import Trainer
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.models.mt5 import MT5Model

from prefix.T5forPrefixGeneration import T5ForPrefixGeneration

__all__ = ["PromptGenerater", "EmbeddingPromptGenerater", "KnowledgePromptGenerater", "PrefixEncoderDecoder"]


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


class PromptGenerater(nn.Module):
    def __init__(self, config, device=None, num_token: int = 5, prefix_dropout: float = 0.5):
        super(PromptGenerater, self).__init__()
        self.num_token = num_token
        self.config = config
        self.device = device
        self.is_knowledge = False
        
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
    
    def forward(self, batch_size: int):
        # encoder
        input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        _, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(batch_size, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # print("prefix_model 131: ", [item.shape for item in past_key_values])
        
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
    
    def save_model(self, path):
        path = os.path.join(path, "prompt_generater.bin")
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        path = os.path.join(path, "prompt_generater.bin")
        self.load_state_dict(torch.load(path))


class EmbeddingPromptGenerater(PromptGenerater):
    def __init__(self, config, device=None, num_token: int = 5, prefix_dropout: float = 0.5):
        super(EmbeddingPromptGenerater, self).__init__(config, device, num_token, prefix_dropout)
        self.num_token = num_token
        
        self.enc_past_kv = nn.Parameter(torch.rand(self.num_token, self.n_layer * 2 * self.n_embd),
                                        requires_grad=True).to(device)
        self.dec_past_kv = nn.Parameter(torch.rand(self.num_token, self.n_layer * 2 * self.n_embd),
                                        requires_grad=True).to(device)
    
    def forward(self, batch_size: int):
        # encoder
        past_key_values = self.enc_past_kv.unsqueeze(dim=0)
        past_key_values = past_key_values.expand(batch_size, self.num_token,
                                                 self.n_layer * 2 * self.n_embd)  # bsz, seqlen, layer*emb
        past_key_values = past_key_values.view(batch_size, self.num_token, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        
        # decoder
        decoder_past_key_values = self.dec_past_kv.unsqueeze(dim=0)
        decoder_past_key_values = decoder_past_key_values.expand(batch_size, self.num_token,
                                                                 self.n_layer * 2 * self.n_embd)
        decoder_past_key_values = decoder_past_key_values.view(batch_size, self.num_token,
                                                               self.match_n_decoder_layer * 2, self.match_n_head,
                                                               self.match_n_embd)
        decoder_past_key_values = self.dropout(decoder_past_key_values)
        decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        
        return past_key_values, decoder_past_key_values
    
    def save_model(self, path):
        path = os.path.join(path, "prompt_generater.bin")
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        path = os.path.join(path, "prompt_generater.bin")
        self.load_state_dict(torch.load(path))


class KnowledgePromptGenerater(PromptGenerater):
    """
    instance + knowledge
    with the unified key/value/query tensors
    """
    def __init__(self, config, device=None, num_token: int = 5, prefix_dropout: float = 0.5, knowledge_file: str = "",
                 do_cross_attention: bool=True):
        super(KnowledgePromptGenerater, self).__init__(config=config, device=device, num_token=num_token,
                                                       prefix_dropout=prefix_dropout)
        self.is_knowledge = True
        self.knowledge_file = knowledge_file
        self.knowledge = None
        self.knowledge_plus = None  # knowledge for arguments
        self.embedding = None
        
        self.tokenized_types = None
        self.tokenized_args = None
        
        self.control_trans = nn.Sequential(
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.num_token * self.n_layer * 2 * self.n_embd)).to(self.device)
        
        self.decoder_control_trans = nn.Sequential(
            nn.Linear(2 * self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.num_token * self.n_decoder_layer * 2 * self.n_embd)).to(self.device)
        
        self.instance_kvq = nn.Linear(self.n_embd, self.mid_dim)
        self.knowledge_kvq = nn.Linear(self.n_embd, self.mid_dim)
        
        self.do_cross_attention = do_cross_attention
    
    def load_knowledge_from_file(self, tokenizer: PreTrainedTokenizer, knowledge_file: str = None):
        """
        load knowledge from schema
        Args:
            knowledge_file ():
            tokenizer (PretrainedTokenizer):
            plm (PreTrainedModel):
        Returns:
        
        """
        if knowledge_file is None:
            knowledge_file = self.knowledge_file
        
        with open(knowledge_file, "r") as file_in:
            lines = [line for line in file_in]
            types = json.loads(lines[0])  # List
            self.tokenized_types = tokenizer(types, return_tensors='pt', padding=True)  # input ids
            
            schemas = json.loads(lines[2])
            schema_list = [" ".join(schemas[t]) for t in types]
            self.tokenized_args = tokenizer(schema_list, return_tensors="pt", padding=True)  # decoder input ids
    
    def encode_knowledge(self, plm):
        input_ids = self.tokenized_types.input_ids.to(self.device)
        # print(type(input_ids), input_ids.device, input_ids.shape)
        attention_mask = self.tokenized_types.attention_mask.to(self.device)
        # print(type(attention_mask), attention_mask.device, attention_mask.shape)
        labels = self.tokenized_args.input_ids.to(self.device)
        # print(type(labels), labels.device, labels.shape)
        labels_attention_mask = self.tokenized_args.attention_mask.to(self.device)
        
        output = plm(input_ids=input_ids,
                     attention_mask=attention_mask,
                     labels=labels)
        
        type_knowledge = output.encoder_last_hidden_state.detach().to(self.device)  # [33, 8, 768]
        self.knowledge = torch.mean(type_knowledge, dim=1).squeeze(dim=1)
    
    def filter_knowledge(self, knowledge, inst: Tensor = None, batch_size: int = 1, knowledge_plus: Tensor = None):
        """
        knowledge filtering method
        Args:
            knowledge (Tensor): knowledge representation
            inst (Tensor): encoder_hidden_states: batch_size * seq_len * embedding_size

        Returns:
        
        """
        
        if inst is None:
            # print("line 277: know_size", knowledge.shape)
            knowledge_kvq = self.knowledge_kvq(knowledge)  # types * mid_dim
            knowledge = torch.mean(knowledge_kvq, dim=0).unsqueeze(dim=0).expand(batch_size, -1)
            return knowledge
        
        inst_len = inst.shape[1]
        
        # print("line 283: inst_size", inst.shape)
        inst = torch.reshape(inst, [-1, inst_len, self.n_embd])
        inst_kvq = self.instance_kvq(inst)  # -1, inst_len, mid_dim
        
        # knowledge: types * embedding
        knowledge = torch.reshape(knowledge, [-1, self.n_embd]).unsqueeze(dim=0).expand(batch_size, -1, self.n_embd)
        knowledge_kvq = self.knowledge_kvq(knowledge)  # -1, know_len, mid_dim
        
        if self.do_cross_attention:
            att_knowledge = torch.mul(
                torch.unsqueeze(inst_kvq, dim=1),
                torch.unsqueeze(knowledge_kvq, dim=2)
            )  # -1, know_len, inst_len, n_embd
            att_knowledge = torch.sum(att_knowledge, dim=2)
            att_knowledge = torch.softmax(att_knowledge, dim=1)  # -1, know_len, n_embd
            knowledge = torch.mul(att_knowledge, knowledge_kvq)
            knowledge = torch.mean(knowledge, dim=1).squeeze(dim=1)

            att_inst = torch.mul(
                torch.unsqueeze(knowledge_kvq, dim=1),
                torch.unsqueeze(inst_kvq, dim=2)
            )  # -1, inst_len, know_len, n_embd
            att_inst = torch.sum(att_inst, dim=2)
            att_inst = torch.softmax(att_inst, dim=1)  # -1, inst, n_embd
            inst = torch.mul(att_inst, inst_kvq)
            inst = torch.mean(inst, dim=1).squeeze(dim=1)
        else:
            knowledge = torch.mean(knowledge_kvq, dim=1).squeeze(dim=1)
            inst = torch.mean(inst_kvq, dim=1).squeeze(dim=1)
            
        return knowledge, inst
    
    def forward(self, batch_size: int, is_decoder: bool = False, encoder_hidden_states=None):
        knowledge = self.knowledge
        if is_decoder:
            assert encoder_hidden_states is not None
            knowledge, inst = self.filter_knowledge(knowledge=knowledge, inst=encoder_hidden_states,
                                                    batch_size=batch_size)  # batch_size * (2*mid_emb)
            # print(f"knowledge shape: {knowledge.shape}")
            # print(f"inst shape: {inst.shape}")
            #
            filtered_knowledge = torch.cat([knowledge, inst], dim=-1)
            decoder_past_key_values = self.decoder_control_trans(filtered_knowledge)
            decoder_past_key_values = decoder_past_key_values.view(batch_size, self.num_token,
                                                                   self.match_n_decoder_layer * 2, self.match_n_head,
                                                                   self.match_n_embd)
            decoder_past_key_values = self.dropout(decoder_past_key_values)
            decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            return decoder_past_key_values
        
        else:
            filtered_knowledge = self.filter_knowledge(knowledge=knowledge, batch_size=batch_size)
            past_key_values = self.control_trans(filtered_knowledge)
            past_key_values = past_key_values.view(batch_size, self.num_token, self.match_n_layer * 2,
                                                   self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            return past_key_values


class KnowledgePromptGeneraterV1(KnowledgePromptGenerater):
    """
    instance + knowledge
    with separated key/value/query tensors
    """
    def __init__(self, config, device=None, num_token: int = 5, prefix_dropout: float = 0.5, knowledge_file: str = "", do_cross_attention: bool=True):
        super(KnowledgePromptGeneraterV1, self).__init__(config=config, device=device, num_token=num_token,
                                                         prefix_dropout=prefix_dropout, do_cross_attention=do_cross_attention)
        self.instance_key = nn.Linear(self.n_embd, self.mid_dim)
        self.instance_value = nn.Linear(self.n_embd, self.mid_dim)
        self.instance_query = nn.Linear(self.n_embd, self.mid_dim)
        
        self.knowledge_key = nn.Linear(self.n_embd, self.mid_dim)
        self.knowledge_value = nn.Linear(self.n_embd, self.mid_dim)
        self.knowledge_query = nn.Linear(self.n_embd, self.mid_dim)
    
    def encode_knowledge(self, plm):
        input_ids = self.tokenized_types.input_ids.to(self.device)
        # print(type(input_ids), input_ids.device, input_ids.shape)
        attention_mask = self.tokenized_types.attention_mask.to(self.device)
        # print(type(attention_mask), attention_mask.device, attention_mask.shape)
        labels = self.tokenized_args.input_ids.to(self.device)
        # print(type(labels), labels.device, labels.shape)
        labels_attention_mask = self.tokenized_args.attention_mask.to(self.device)
        output = plm(input_ids=input_ids, attention_mask=attention_mask,
                     labels=labels)
        
        knowledge = output.encoder_last_hidden_state.detach()  # [33, 8, 768]
        self.knowledge = torch.mean(knowledge, dim=1).squeeze()
    
    def filter_knowledge(self, knowledge, knowledge_plus: Tensor = None, inst: Tensor = None, batch_size: int = 1):
        """
        knowledge filtering method
        Args:
            knowledge (Tensor): knowledge representation
            inst (Tensor): encoder_hidden_states: batch_size * seq_len * embedding_size

        Returns:

        """
        if inst is None:
            # print("line 277: know_size", knowledge.shape)
            knowledge_value = self.knowledge_value(knowledge)  # types * mid_dim
            knowledge = torch.mean(knowledge_value, dim=0).unsqueeze(dim=0).expand(batch_size, -1)
            return knowledge
        
        inst_len = inst.shape[1]
        
        # print("line 283: inst_size", inst.shape)
        inst = torch.reshape(inst, [-1, inst_len, self.n_embd])
        inst_key = self.instance_key(inst)  # -1, inst_len, mid_dim
        inst_value = self.instance_value(inst)  # -1, inst_len, mid_dim
        inst_query = self.instance_query(inst)  # -1, inst_len, mid_dim
        
        # knowledge: types * embedding
        knowledge = torch.reshape(knowledge, [-1, self.n_embd]).unsqueeze(dim=0).expand(batch_size, -1, self.n_embd)
        knowledge_key = self.knowledge_key(knowledge)  # -1, know_len, mid_dim
        knowledge_value = self.knowledge_value(knowledge)  # -1, know_len, mid_dim
        knowledge_query = self.knowledge_key(knowledge)  # -1, know_len, mid_dim
        
        if self.do_cross_attention:
            att_knowledge = torch.mul(
                torch.unsqueeze(inst_query, dim=1),
                torch.unsqueeze(knowledge_key, dim=2)
            )  # -1, know_len, inst_len, n_embd
            att_knowledge = torch.sum(att_knowledge, dim=2)
            att_knowledge = torch.softmax(att_knowledge, dim=1)  # -1, know_len, n_embd
            knowledge = torch.mul(att_knowledge, knowledge_value)
            knowledge = torch.mean(knowledge, dim=1).squeeze(dim=1)
    
            att_inst = torch.mul(
                torch.unsqueeze(knowledge_query, dim=1),
                torch.unsqueeze(inst_key, dim=2)
            )  # -1, inst_len, know_len, n_embd
            att_inst = torch.sum(att_inst, dim=2)
            att_inst = torch.softmax(att_inst, dim=1)  # -1, inst, n_embd
            inst = torch.mul(att_inst, inst_value)
            inst = torch.mean(inst, dim=1).squeeze(dim=1)
        else:
            knowledge = torch.mean(knowledge_value, dim=1).squeeze(dim=1)
            inst = torch.mean(inst_value, dim=1).squeeze(dim=1)
        
        return knowledge, inst


class KnowledgePromptGeneraterV2(KnowledgePromptGeneraterV1):
    """
    instance + knowledge + pseudo tokens
    """
    def __init__(self, config, device=None, num_token: int = 5, prefix_dropout: float = 0.5, knowledge_file: str = "",
                 do_cross_attention: bool=True):
        super(KnowledgePromptGeneraterV2, self).__init__(config=config, device=device, num_token=num_token,
                                                         prefix_dropout=prefix_dropout, do_cross_attention=do_cross_attention)

        self.control_trans = nn.Sequential(
            nn.Linear(2*self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.num_token * self.n_layer * 2 * self.n_embd)).to(self.device)

        self.decoder_control_trans = nn.Sequential(
            nn.Linear(3 * self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.num_token * self.n_decoder_layer * 2 * self.n_embd)).to(self.device)
        
        self.task_conditioning = nn.Parameter(torch.rand(self.mid_dim), requires_grad=True).to(self.device)
        self.decoder_task_conditioning = nn.Parameter(torch.rand(self.mid_dim), requires_grad=True).to(self.device)

    def forward(self, batch_size: int, is_decoder: bool = False, encoder_hidden_states=None):
        knowledge = self.knowledge
        if is_decoder:
            assert encoder_hidden_states is not None
            knowledge, inst = self.filter_knowledge(knowledge=knowledge, inst=encoder_hidden_states,
                                                    batch_size=batch_size)  # batch_size * (2*mid_emb)
            # print(f"knowledge shape: {knowledge.shape}")
            # print(f"inst shape: {inst.shape}")
            #
            decoder_task_condition = self.decoder_task_conditioning.unsqueeze(dim=0).expand([batch_size, self.mid_dim])
            
            filtered_knowledge = torch.cat([knowledge, inst, decoder_task_condition], dim=-1)
            decoder_past_key_values = self.decoder_control_trans(filtered_knowledge)
            decoder_past_key_values = decoder_past_key_values.view(batch_size, self.num_token,
                                                                   self.match_n_decoder_layer * 2, self.match_n_head,
                                                                   self.match_n_embd)
            decoder_past_key_values = self.dropout(decoder_past_key_values)
            decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            return decoder_past_key_values
    
        else:
            task_condition = self.task_conditioning.unsqueeze(dim=0).expand([batch_size, self.mid_dim])
            filtered_knowledge = self.filter_knowledge(knowledge=knowledge, batch_size=batch_size)
            filtered_knowledge = torch.cat([filtered_knowledge, task_condition], dim=-1)
    
            past_key_values = self.control_trans(filtered_knowledge)
            past_key_values = past_key_values.view(batch_size, self.num_token, self.match_n_layer * 2,
                                                   self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            return past_key_values


class HybridPromptGenerater(KnowledgePromptGenerater):
    def __init__(self, config, device=None, num_token: int = 5, prefix_dropout: float = 0.5, knowledge_file: str = ""):
        super(HybridPromptGenerater, self).__init__(config=config, device=device, num_token=num_token,
                                                    prefix_dropout=prefix_dropout,
                                                    knowledge_file=knowledge_file)
        self.input_tokens = nn.Parameter(torch.arange(self.num_token - 1).long(),
                                         requires_grad=False).to(self.device)
        
        self.wte = nn.Embedding(self.num_token - 1, self.n_embd).to(self.device)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_layer * 2 * self.n_embd)).to(self.device)
        
        self.decoder_input_tokens = nn.Parameter(torch.arange(self.num_token - 2).long(),
                                                 requires_grad=False).to(self.device)
        
        self.decoder_wte = nn.Embedding(self.num_token - 2, self.n_embd).to(self.device)
        self.decoder_control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_decoder_layer * 2 * self.n_embd)).to(self.device)
        
        self.instance_kvq = nn.Linear(self.n_embd, self.n_embd)
        self.knowledge_kvq = nn.Linear(self.n_embd, self.n_embd)
    
    def forward(self, batch_size: int, is_decoder: bool = False, encoder_hidden_states=None):
        knowledge = self.knowledge
        if is_decoder:
            inst = encoder_hidden_states
            filtered_knowledge = self.filter_knowledge(knowledge=knowledge, inst=inst, batch_size=batch_size)
            filtered_knowledge = torch.stack(filtered_knowledge, dim=1)
            # print(filtered_knowledge.shape)
            decoder_input_tokens = self.decoder_input_tokens.unsqueeze(0).expand(batch_size, -1)
            
            # batch_size * 1 * embed_size
            temp_control = self.decoder_wte(decoder_input_tokens)
            # print(temp_control.shape)
            temp_control = torch.cat([filtered_knowledge, temp_control], dim=1)
            # print(f"temp_control size: {temp_control.shape}")
            
            decoder_past_key_values = self.decoder_control_trans(temp_control)
            decoder_past_key_values = decoder_past_key_values.view(batch_size, self.num_token,
                                                                   self.match_n_decoder_layer * 2, self.match_n_head,
                                                                   self.match_n_embd)
            
            decoder_past_key_values = self.dropout(decoder_past_key_values)
            decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            return decoder_past_key_values
        
        else:
            filtered_knowledge = self.filter_knowledge(knowledge=knowledge, batch_size=batch_size).unsqueeze(dim=1)
            # print(filtered_knowledge.shape)
            input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
            temp_control = self.wte(input_tokens)
            # print(temp_control.shape)
            temp_control = torch.cat([filtered_knowledge, temp_control], dim=1)
            
            past_key_values = self.control_trans(temp_control)
            # print(past_key_values.shape)
            past_key_values = past_key_values.view(batch_size, self.num_token, self.match_n_layer * 2,
                                                   self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            return past_key_values


class HybridPromptGeneraterPlus(KnowledgePromptGenerater):
    def __init__(self, config, device=None, num_token: int = 5, prefix_dropout: float = 0.5, knowledge_file: str = ""):
        super(HybridPromptGeneraterPlus, self).__init__(config=config, device=device, num_token=num_token,
                                                        prefix_dropout=prefix_dropout,
                                                        knowledge_file=knowledge_file)
        self.input_tokens = nn.Parameter(torch.arange(self.num_token - 2).long(),
                                         requires_grad=False).to(self.device)
        
        self.wte = nn.Embedding(self.num_token - 2, self.n_embd).to(self.device)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_layer * 2 * self.n_embd)).to(self.device)
        
        self.decoder_input_tokens = nn.Parameter(torch.arange(self.num_token - 3).long(),
                                                 requires_grad=False).to(self.device)
        
        self.decoder_wte = nn.Embedding(self.num_token - 3, self.n_embd).to(self.device)
        self.decoder_control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_decoder_layer * 2 * self.n_embd)).to(self.device)
        
        self.instance_kvq = nn.Linear(self.n_embd, self.n_embd)
        self.knowledge_kvq = nn.Linear(self.n_embd, self.n_embd)
        self.knowledge_plus_kvq = nn.Linear(self.n_embd, self.n_embd)
    
    def forward(self, batch_size: int, is_decoder: bool = False, encoder_hidden_states=None):
        knowledge = self.knowledge
        knowledge_plus = self.knowledge_plus
        if is_decoder:
            inst = encoder_hidden_states
            filtered_knowledge = self.filter_knowledge(knowledge, knowledge_plus=knowledge_plus, inst=inst,
                                                       batch_size=batch_size)
            filtered_knowledge = torch.stack(filtered_knowledge, dim=1)
            print(filtered_knowledge.shape)
            decoder_input_tokens = self.decoder_input_tokens.unsqueeze(0).expand(batch_size, -1)
            
            # batch_size * 1 * embed_size
            temp_control = self.decoder_wte(decoder_input_tokens)
            print(temp_control.shape)
            temp_control = torch.cat([filtered_knowledge, temp_control], dim=1)
            print(f"temp_control size: {temp_control.shape}")
            
            decoder_past_key_values = self.decoder_control_trans(temp_control)
            decoder_past_key_values = decoder_past_key_values.view(batch_size, self.num_token,
                                                                   self.match_n_decoder_layer * 2, self.match_n_head,
                                                                   self.match_n_embd)
            
            decoder_past_key_values = self.dropout(decoder_past_key_values)
            decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            return decoder_past_key_values
        
        else:
            filtered_knowledge = self.filter_knowledge(knowledge=knowledge, knowledge_plus=knowledge_plus,
                                                       batch_size=batch_size)
            filtered_knowledge = torch.stack(filtered_knowledge, dim=1)
            # print(filtered_knowledge.shape)
            input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
            temp_control = self.wte(input_tokens)
            # print(temp_control.shape)
            temp_control = torch.cat([filtered_knowledge, temp_control], dim=1)
            
            past_key_values = self.control_trans(temp_control)
            # print(past_key_values.shape)
            past_key_values = past_key_values.view(batch_size, self.num_token, self.match_n_layer * 2,
                                                   self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            return past_key_values
    
    def encode_knowledge(self, plm):
        types = self.tokenized_types.input_ids.to(self.device)
        # print(type(input_ids), input_ids.device, input_ids.shape)
        types_attention_mask = self.tokenized_types.attention_mask.to(self.device)
        # print(type(attention_mask), attention_mask.device, attention_mask.shape)
        arguments = self.tokenized_args.input_ids.to(self.device)
        # print(type(labels), labels.device, labels.shape)
        arguments_attention_mask = self.tokenized_args.attention_mask.to(self.device)
        
        output = plm(input_ids=types,
                     attention_mask=types_attention_mask,
                     labels=arguments)
        type_knowledge = output.encoder_last_hidden_state.detach().to(self.device)  # [33, 8, 768]
        self.knowledge = torch.mean(type_knowledge, dim=1).squeeze(dim=1)
        
        output = plm(input_ids=arguments,
                     attention_mask=arguments_attention_mask,
                     labels=types)
        arguments_knowledge = output.encoder_last_hidden_state.detach().to(self.device)  # [33, 8, 768]
        print(f"argument knowledge: {arguments_knowledge.shape}")
        self.knowledge_plus = torch.mean(arguments_knowledge, dim=1).squeeze(dim=1)
        print(f"argument knowledge: {self.knowledge_plus.shape}")
    
    def filter_knowledge(self, knowledge, knowledge_plus: Tensor = None, inst: Tensor = None, batch_size: int = 1):
        """
        knowledge filtering method
        Args:
            knowledge (Tensor): knowledge representation
            knowledge_plus (Tensor): knowledge for arguments
            inst (Tensor): encoder_hidden_states: batch_size * seq_len * embedding_size
        Returns:

        """
        
        if inst is None:
            # print("line 277: know_size", knowledge.shape)
            knowledge_kvq = self.knowledge_kvq(knowledge)  # types * mid_dim
            knowledge = torch.mean(knowledge_kvq, dim=0).unsqueeze(dim=0).expand(batch_size, -1)
            knowledge_plus_kvq = self.knowledge_plus_kvq(knowledge)
            knowledge_plus = torch.mean(knowledge_plus_kvq, dim=0).unsqueeze(dim=0).expand(batch_size, -1)
            return knowledge, knowledge_plus
        
        inst_len = inst.shape[1]
        
        # print("line 283: inst_size", inst.shape)
        inst = torch.reshape(inst, [-1, inst_len, self.n_embd])
        inst_kvq = self.instance_kvq(inst)  # -1, inst_len, mid_dim
        
        # knowledge: types * embedding
        knowledge = torch.reshape(knowledge, [-1, self.n_embd]).unsqueeze(dim=0).expand(batch_size, -1, self.n_embd)
        knowledge_kvq = self.knowledge_kvq(knowledge)  # -1, know_len, mid_dim
        
        att_knowledge = torch.mul(
            torch.unsqueeze(inst_kvq, dim=1),
            torch.unsqueeze(knowledge_kvq, dim=2)
        )  # -1, know_len, inst_len, n_embd
        att_knowledge = torch.sum(att_knowledge, dim=2)
        att_knowledge = torch.softmax(att_knowledge, dim=1)  # -1, know_len, n_embd
        knowledge = torch.mul(att_knowledge, knowledge_kvq)
        knowledge = torch.mean(knowledge, dim=1).squeeze(dim=1)
        
        knowledge_plus = torch.reshape(knowledge_plus, [-1, self.n_embd]).unsqueeze(dim=0).expand(batch_size, -1,
                                                                                                  self.n_embd)
        knowledge_plus_kvq = self.knowledge_kvq(knowledge_plus)  # -1, know_len, mid_dim
        print(f"knowledge_plus_kvq: {knowledge_plus_kvq.shape}")
        print(f"knowledge_attention: {att_knowledge.shape}")
        knowledge_plus = torch.mul(att_knowledge, knowledge_plus_kvq)
        knowledge_plus = torch.mean(knowledge_plus, dim=1).squeeze(dim=1)
        
        att_inst = torch.mul(
            torch.unsqueeze(knowledge_kvq, dim=1),
            torch.unsqueeze(inst_kvq, dim=2)
        )  # -1, inst_len, know_len, n_embd
        att_inst = torch.sum(att_inst, dim=2)
        att_inst = torch.softmax(att_inst, dim=1)  # -1, inst, n_embd
        inst = torch.mul(att_inst, inst_kvq)
        inst = torch.mean(inst, dim=1).squeeze(dim=1)
        
        return knowledge, knowledge_plus, inst


class AdapterGenerater(PromptGenerater):
    def __init__(self, config):
        super(AdapterGenerater, self).__init__(config)
        
        self.adapter_module = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_embd)
        )
        
        self.encoder_adapter_module_list = nn.ModuleList(
            [copy.deepcopy(self.adapter_module) for i in range(self.n_layer)]
        )
        
        self.decoder_adapter_module_list = nn.ModuleList(
            [copy.deepcopy(self.adapter_module) for i in range(self.n_decoder_layer)]
        )


class PrefixEncoderDecoder(nn.Module):
    def __init__(self, model: PreTrainedModel, prompt_generater: PromptGenerater, training_args: Namespace):
        super(PrefixEncoderDecoder, self).__init__()
        self.module = model
        self.prompt_generater = prompt_generater
        self.config = model.config
        self.training_args = training_args
        self.device = training_args.device
        self.tuning_type = training_args.tuning_type
        self.is_knowledge = prompt_generater.is_knowledge
        self._keys_to_ignore_on_save = None
        
        self.is_encoder_conditioning = training_args.is_encoder_conditioning
        self.is_decoder_conditioning = training_args.is_decoder_conditioning
        
        self.num_token = self.prompt_generater.num_token
        
        self.is_modified_model = False
        
        if "adapter" in self.tuning_type:
            self.modify_model_adapter()
        else:
            self.modify_model()
    
    def forward(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        
        if "adapter" in self.tuning_type:
            outputs = self.module(**kwargs)
        else:
            if self.is_knowledge:
                self.prompt_generater.encode_knowledge(plm=self.module)
                prefix_past_key_values = self.prompt_generater(batch_size=batch_size, is_decoder=False)
                kwargs["prefix_key_values"] = [prefix_past_key_values]
                # print(f"is_knowledge 695: {self.is_knowledge}")
            else:
                prefix_past_key_values = self.prompt_generater(batch_size)
                kwargs["prefix_key_values"] = prefix_past_key_values
            outputs = self.module(**kwargs)
        
        return outputs
    
    def modify_model(self):
        if self.is_modified_model:
            return None
        if self.tuning_type == "prefix":
            for pp in self.module.parameters():
                pp.requires_grad = False
        
        if isinstance(self.module, T5ForPrefixGeneration):
            
            # for encoder
            backup_encoder_forward_functions = []
            for i, layer_module in enumerate(self.module.encoder.block):
                backup_encoder_forward_functions.append(layer_module.layer[0].forward)
                
                def modified_encoder_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    if kwargs["prefix_key_values"] is not None:
                        if kwargs['past_key_value'] is None:
                            kwargs['past_key_value'] = kwargs["prefix_key_values"][0][layer_id]
                        
                        if kwargs['attention_mask'] is not None:
                            am = kwargs['attention_mask']
                            kwargs['attention_mask'] = torch.cat(
                                [torch.ones((*am.shape[:-1], self.num_token), dtype=am.dtype, device=am.device), am],
                                dim=-1)
                        # print(kwargs['past_key_value'].shape)
                    return backup_encoder_forward_functions[layer_id](*args, **kwargs)
                
                if self.is_encoder_conditioning:
                    layer_module.layer[0].forward = partial(modified_encoder_forward, layer_id=i)
            
            # for knowledge aware prefix
            if self.is_knowledge:
                backup_decoder_forward_function = self.module.decoder.forward
                
                def modified_decoder_forward(*args, **kwargs):
                    is_decoder = kwargs.pop("is_decoder")
                    if kwargs["prefix_key_values"] is not None:
                        encoder_hidden_states = kwargs["encoder_hidden_states"]
                        batch_size = kwargs["input_ids"].shape[0]
                        kwargs["prefix_key_values"].append(self.prompt_generater(is_decoder=is_decoder,
                                                                                 batch_size=batch_size,
                                                                                 encoder_hidden_states=encoder_hidden_states))
                    return backup_decoder_forward_function(*args, **kwargs)
                
                self.module.decoder.forward = partial(modified_decoder_forward, is_decoder=True)
            
            # for decoder
            backup_decoder_self_attn_forward_functions = []
            backup_decoder_cross_attn_forward_functions = []
            for i, layer_module in enumerate(self.module.decoder.block):
                backup_decoder_self_attn_forward_functions.append(layer_module.layer[0].forward)
                
                def modified_decoder_self_attn_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    if kwargs["prefix_key_values"] is not None:
                        
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
                
                if self.is_decoder_conditioning:
                    layer_module.layer[1].forward = partial(modified_decoder_cross_attn_forward, layer_id=i)
            
            self.backup_encoder_forward_functions = backup_encoder_forward_functions
            self.backup_decoder_self_attn_forward_functions = backup_decoder_self_attn_forward_functions
            self.backup_decoder_cross_attn_forward_functions = backup_decoder_cross_attn_forward_functions
    
    def modify_model_adapter(self):
        if self.is_modified_model:
            return None
        if self.tuning_type == "adapter":  # otherwise both_adapter
            for pp in self.module.parameters():
                pp.requires_grad = False
        
        if isinstance(self.module, T5ForPrefixGeneration):
            
            # for encoder
            backup_encoder_forward_functions = []
            for i, layer_module in enumerate(self.module.encoder.block):
                backup_encoder_forward_functions.append(layer_module.layer[0].forward)
                
                def modified_encoder_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    layer_output = backup_encoder_forward_functions[layer_id](*args, **kwargs)
                    
                    hidden_states = layer_output[0]
                    adapted_hidden_states = self.prompt_generater.encoder_adapter_module_list[i](hidden_states)
                    hidden_states = torch.add(hidden_states, adapted_hidden_states)
                    layer_output = list(layer_output)
                    layer_output[0] = hidden_states
                    layer_output = tuple(layer_output)
                    return layer_output
                
                layer_module.layer[0].forward = partial(modified_encoder_forward, layer_id=i)
            
            # for decoder
            backup_decoder_self_attn_forward_functions = []
            backup_decoder_cross_attn_forward_functions = []
            for i, layer_module in enumerate(self.module.decoder.block):
                backup_decoder_self_attn_forward_functions.append(layer_module.layer[0].forward)
                
                def modified_decoder_self_attn_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    layer_output = backup_decoder_self_attn_forward_functions[layer_id](*args, **kwargs)
                    
                    hidden_states = layer_output[0]
                    adapted_hidden_states = self.prompt_generater.encoder_adapter_module_list[i](hidden_states)
                    hidden_states = torch.add(hidden_states, adapted_hidden_states)
                    layer_output = list(layer_output)
                    layer_output[0] = hidden_states
                    layer_output = tuple(layer_output)
                    return layer_output
                
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
        if "adapter" in self.tuning_type:
            return self.module.generate(*args, **kwargs)
        else:
            if self.is_knowledge:
                self.prompt_generater.encode_knowledge(plm=self.module)
                prefix_past_key_values = self.prompt_generater(batch_size=batch_size, is_decoder=False)
                kwargs["prefix_key_values"] = [prefix_past_key_values]
            else:
                prefix_past_key_values = self.prompt_generater(batch_size)
                kwargs["prefix_key_values"] = prefix_past_key_values
            
            return self.module.generate(*args, **kwargs)

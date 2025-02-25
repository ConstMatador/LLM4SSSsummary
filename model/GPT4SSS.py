import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
import math
import logging

from utils.conf import Configuration
from model.embed import DataEmbedding


class GPT4SSS(nn.Module):
    def __init__(self, conf:Configuration):
        super(GPT4SSS, self).__init__()
        
        self.conf = conf
        self.device = self.conf.getEntry('device')
        self.llm_path = self.conf.getEntry('llm_path')
        self.llm_type = self.conf.getEntry('llm_type')
        self.llm_layers = self.conf.getEntry('llm_layers')
        self.dim_llm = self.conf.getEntry('dim_llm')
        self.dim_ff = self.conf.getEntry('dim_ff')
        self.len_series = self.conf.getEntry('len_series')
        self.len_reduce = self.conf.getEntry('len_reduce')
        self.llm_pos = self.llm_path + self.llm_type + '/'
        
        if self.llm_type == 'gpt2':
            self.llm = GPT2Model.from_pretrained(self.llm_pos)  # pyright: ignore
        elif self.llm_type == 'bert':
            self.llm = BertModel.from_pretrained(self.llm_pos)  # pyright: ignore
        elif self.llm_type == 'llama7b':
            self.llm = LlamaModel.from_pretrained(self.llm_pos) # pyright: ignore
            
        self.enc_embedding = DataEmbedding(self.conf)
        
        logging.info('Layer Unfrozen : ')
        for _, (name, param) in enumerate(self.llm.named_parameters()):
            if 'ln' in name or 'wpe' in name or 'mlp' in name:
                param.requires_grad = True
                logging.info(f'{name}')
            else:
                param.requires_grad = False
        
        self.flatten = nn.Flatten(start_dim=1)
        self.dim_mid = math.floor(math.sqrt(self.len_series * self.dim_ff))
        self.linear1 = nn.Linear(self.len_series * self.dim_ff, self.dim_mid)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(self.dim_mid, self.len_reduce)
        
        
    def forward(self, x):
        # x: [batch_size, len_series]
        x = self.enc_embedding(x)
        # x: [batch_size, len_series, dim_model]
        x = nn.functional.pad(x, (0, self.dim_llm - x.size(-1)))
        # x: [batch_size, len_series, dim_llm]
        x = self.llm(inputs_embeds=x).last_hidden_state
        # x: [batch_size, len_series, dim_llm]
        x = x[:, :, :self.dim_ff]
        # x: [batch_size, len_series, dim_ff]
        x = self.flatten(x)
        # x: [batch_size, len_series * dim_ff]
        x = self.linear1(x)
        # x: [batch_size, dim_mid]
        x = self.tanh(x)
        # x: [batch_size, dim_mid]
        x = self.dropout(x)
        # x: [batch_size, dim_mid]
        x = self.linear2(x)
        # x: [batch_size, len_reduce]
        
        return x
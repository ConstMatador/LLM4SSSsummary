import numpy as np
import torch
from torch import nn
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from einops import rearrange
import pandas as pd
import logging

from utils.conf import Configuration
from model.prompt import Prompt


class S2IPLLM(nn.Module):
    def __init__(self, conf: Configuration):
        super(S2IPLLM, self).__init__()
        
        self.conf = conf
        self.device = self.conf.getEntry('device')
        self.batch_size = self.conf.getEntry('batch_size')
        self.llm_path = self.conf.getEntry('llm_path')
        self.llm_type = self.conf.getEntry('llm_type')
        self.llm_layers = self.conf.getEntry('llm_layers')
        self.dim_llm = self.conf.getEntry('dim_llm')
        self.len_series = self.conf.getEntry('len_series')
        self.len_reduce = self.conf.getEntry('len_reduce')
        self.llm_pos = self.llm_path + self.llm_type + '/'
        
        self.dim_llm = conf.getEntry('dim_llm')
        self.win_size = conf.getEntry('win_size')
        self.stride = conf.getEntry('stride')
        
        self.trend_len = conf.getEntry('trend_len')
        self.seasonal_len = conf.getEntry('seasonal_len')
        
        self.top_k = conf.getEntry('top_k')

        if self.llm_type == 'gpt2':
            self.llm = GPT2Model.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore
        elif self.llm_type == 'bert':
            self.llm = BertModel.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore
        elif self.llm_type == 'llama7b':
            self.llm = llamaModel.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore
            
        for _, (name, param) in enumerate(self.llm.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
                logging.info(f'{name}')
            else:
                param.requires_grad = False    
            
        self.win_num = (self.len_series - self.win_size) // self.stride + 1
        
        self.inlayer = nn.Linear(self.win_size*3, self.dim_llm)
        
        self.prompt_pool = Prompt(self.conf, self.llm.wte.weight)
        
        self.flatten = nn.Flatten(start_dim=-2)
        
        self.outlayer = nn.Linear((self.win_num + self.top_k) * self.dim_llm, self.len_reduce)
        

    def forward(self, x):
        # x:[batch_size, len_series]
        
        # function decompose
        def decompose(x):
            df = pd.DataFrame(x)
            trend = df.rolling(window=self.trend_len, center=True).mean().fillna(method='bfill').fillna(method='ffill')      #pyright:ignore
            detrended = df - trend
            seasonal = detrended.groupby(detrended.index % self.seasonal_len).transform('mean').fillna(method='bfill').fillna(method='ffill') 
            residuals = df - trend - seasonal
            combined = np.stack([trend, seasonal, residuals], axis=1)
            return combined
        
        x = np.apply_along_axis(decompose, 1, x.cpu().numpy())
        # x:[batch_size, len_series, 3]
        x = torch.tensor(x).to(self.device)
        x = x.reshape(-1, self.len_series, 3)
        # x:[batch_size, len_series, 3]
        x = rearrange(x, 'b l c -> b c l')
        # x:[batch_size, 3, len_series]
        x = x.unfold(dimension=-1, size=self.win_size, step=self.stride)
        # x:[batch_size, 3, win_num, win_size]
        x = rearrange(x, 'b c n s -> b n (s c)')
        # x:[batch_size, win_num, win_size*3]
        x = self.inlayer(x.float())
        # x:[batch_size, win_num, dim_llm]
        outs = self.prompt_pool(x)
        prompted_embedding = outs['prompted_embedding']
        # prompted_embedding:[batch_size, win_num + top_k, dim_llm]
        output_embedding = self.llm(inputs_embeds = prompted_embedding).last_hidden_state
        # output_embedding:[batch_size, win_num + top_k, dim_llm]
        x = self.flatten(output_embedding)
        # x:[batch_size, (win_num + top_k)*dim_llm]
        x = self.outlayer(x)
        # x:[batch_size, len_reduce]
        return x
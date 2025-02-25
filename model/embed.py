import torch
import torch.nn as nn
import math

from utils.conf import Configuration


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False     # pyright: ignore

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class DataEmbedding(nn.Module):
    def __init__(self, conf:Configuration):
        super(DataEmbedding, self).__init__()
        
        self.dim_model = conf.getEntry('dim_model')
        self.dropout = conf.getEntry('dropout')
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=1, out_channels=self.dim_model, kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.position_mbedding = PositionalEmbedding(self.dim_model)
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        # x: [batch_size, len_series]
        x = x.unsqueeze(1)
        # x: [batch_size, 1, len_series]
        x = self.tokenConv(x).permute(0, 2, 1)
        # x: [batch_size, len_series, dim_model]
        x = x + self.position_mbedding(x)
        # x: [batch_size, len_series, dim_model]
        
        return self.dropout(x)
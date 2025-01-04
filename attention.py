import torch 
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) ->None:
        super().__init__()
        self.d_model = d_model 
        self.h = h 
        
        assert d_model % h == 0, "d_model is not divisible by h"
    
        #d_k = d_model / h

        self.d_k = d_model // h 

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
    
    def forward(self, q, k , v, mask):
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)










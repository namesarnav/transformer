import torch 
import torch.nn as nn
import math
from . import * 

class MultiHeadAttention(nn.Module):

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
    
    def forward(self, x):
        










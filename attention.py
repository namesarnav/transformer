import torch 
import torch.nn as nn
import math
from . import * 

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, seq_len: int) ->:
        super().__init__()
        self.d_model = d_model 
        self.seq_
import torch 
from torch import nn 
import math
from . import *

# Forward Pass
class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int , dropout: float) -> None:

        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #w1 & b1 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #w2 b2
        
    def forward(self, x): 
         return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

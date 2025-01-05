from attention import *
from model import * 

import torch 
import torch.nn as nn

import math


class Encoder(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward: FeedForward, dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward
        self.residual_connections  = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    


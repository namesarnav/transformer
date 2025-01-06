from attention import *
from model import * 

import torch 
import torch.nn as nn

import math

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward: FeedForward, dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward
        self.residual_connections  = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask ): 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm =  LayerNormalize()

    def forward(self, x , mask): 
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
    
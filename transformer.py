from model import *
from attention import *
import torch 
import torch.nn as nn 


class Transformer(nn.Module): 

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding,
                  tgt_embed: InputEmbedding, src_pos: PositionalEncoding, 
                  tgt_pos: PositionalEncoding, proj_layer: ProjectionLayer):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.proj = proj_layer
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos

    def encode(self, src, src_mask ):
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask): 
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decode(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.proj_layer(x)
    


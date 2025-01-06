import torch 
from torch import nn
from math import sqrt, log
from attention import * 
class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self. d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, seq_len: int ):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.seq_len = seq_len

        #matrix of shape seq_len x d_model  
        pe = torch.zeros(seq_len, d_model )   

        position = torch.arrange(0, seq_len, dtype = torch.float).unsqueeze(1) ## Numerator term 
        div_term = torch.exp(torch.arrange(0, d_model, 2).float() * (-log(10000.0)/ d_model) ) #divisor term

        #Applying the trig functions to calc pos vector

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
 
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe ) 

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1]], : ).requires_grad_(False)


# Forward Pass
class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int , dropout: float) -> None:

        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #w1 & b1 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #w2 b2
        
    def forward(self, x): 
         return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
class ResidualConnection(nn.Module): 
    
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.norm = LayerNormalize()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNormalize(nn.Module):

    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):

        mean = x.mean(dim = -1 , keepDim =True)
        std = x.std(dim=-1 , keepDim = True)

        return self.alpha * (x-mean) / (std + self.eps) + self.bias


"""
Encoder Block
"""

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
    
    


"""Decoder Block
"""

class DecoderBlock(nn.Module): 
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward: FeedForward, dropout: float):
        super().__init__() 
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward = feed_forward
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(3)])
 
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) 
        x = self.residual_connections[2](x, self.feed_forward)
        return x 

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__() 
        self.layers = layers
        self.norm = LayerNormalize()
    
    def forward(self, x, encoder_output, src_mask , tgt_mask):
        
        for layer in self.layers: 
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)

class ProjectionLayer(nn.Module):
    '''
    Projection layer for final conversion of the output matrix to the seq len
    '''
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__() 
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1 )
    
    
### Building the transformer


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                      src_seq_len: int, tgt_seq_len: int, d_model: int, 
                      N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048):
    '''
    Build the final transformer
    '''
    
    # ----- Embedding Layer ---- #

    # Inp
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    #Pos
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N): 
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)

        encoder_block  = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_block.append(encoder_block)

    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)

        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,
                                      feed_forward_block, dropout)

        decoder_blocks.append(decoder_block)


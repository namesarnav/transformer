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

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout ): 
        d_k = query.shape[-1]
        
        attention_scores = (query @ key.transpose[-2, -1]) / math.sqrt(d_k)
        if mask is not None: 
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        
        if dropout is not None: 
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value ) , attention_scores
    
    def forward(self, q, k , v, mask):
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, h , seq_len d_k) ---> 
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1 , self.h * self.d_k)
    
        return self.w_o(x)
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
    
class EncoderBlock(nn.Module):
    '''
    Encoder
    '''
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, 
                 feed_forward: FeedForward, dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward
        self.residual_connections  = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask ): 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    '''
    Construct the encoder block
    '''
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm =  LayerNormalize()

    def forward(self, x , mask): 
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module): 
    '''
    Decoder Block
    '''
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, 
                 cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward: FeedForward, dropout: float):
        

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
    '''
    Construct the decoder block
    '''
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

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                      src_seq_len: int, tgt_seq_len: int, d_model: int, 
                      N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048):
    
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
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

        decoder_block = DecoderBlock(decoder_self_attention_block,
                                     decoder_cross_attention_block,
                                    feed_forward_block, dropout)

        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

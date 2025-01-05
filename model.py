import torch 
from torch import nn
from math import sqrt, log

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



#-----------#----------#-----
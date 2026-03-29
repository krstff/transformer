import torch.nn as nn
import torch
import math
from einops import rearrange
import config

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # expand dimension
        self.d_exp    = nn.Linear(embed_dim, 4 * embed_dim)
        self.act     = nn.GELU()
        # reduce dimension back
        self.d_red  = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(config.RESID_DROP)
    
    def forward(self, x):
        x = self.d_exp(x)
        x = self.act(x)
        x = self.d_red(x)
        x = self.dropout(x)
        return x

class MSMAttention(nn.Module):
    # https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/
    # https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

    def __init__(self, input_dim, num_embed, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = num_embed // num_heads

        # Create mask for masked attention, registered as buffer as this is not a learnable parameter
        self.register_buffer("mask", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE))
                                     .view(1, 1, config.BLOCK_SIZE, config.BLOCK_SIZE))

        # Projection for calculating Q K V
        # could also be three different projections (=> num_embed * 3)
        self.qkv = nn.Linear(input_dim, num_embed * 3)
        self.linear = nn.Linear(input_dim, num_embed)

        # Dropout for the attention layer (is disabled during model.Eval())
        self.a_dropout = nn.Dropout(config.ATTENTION_DROP)
        self.resid_dropout = nn.Dropout(config.RESID_DROP)
    
    def forward(self, x):
        # batch size, sequence length, embedding size (equal to input_dim)
        B, T, C = x.size()

        qkv = self.qkv(x)
        # split into heads
        qkv = rearrange(qkv, 'b t (three h d) -> b t h (three d)', three=3, h=self.num_heads)
        # transpose for dimension matching
        qkv = rearrange(qkv, 'b t h (three d) -> b h t (three d)', three=3)

        # split into q k v
        # each of dimension (B, num_heads, T, head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        e = q @ k.transpose(-1, -2)
        e = e / math.sqrt(k.size(-1))

        # Mask the lower triangle and compute softmax
        e = e.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        a = e.softmax(dim=-1)
        a = self.a_dropout(a)

        y = a @ v
        # merge heads together
        y = y.rearrange(y, 'b h t d -> b t (h d)')

        y = self.linear(y)
        y = self.resid_dropout(y)

        return y

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):        
        # Calculate mean and variance across the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # normalize and apply scale/shift
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias






    
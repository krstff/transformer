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

    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Create mask for masked attention, registered as buffer as this is not a learnable parameter
        self.register_buffer("mask", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE))
                                     .view(1, 1, config.BLOCK_SIZE, config.BLOCK_SIZE))

        # Projection for calculating Q K V
        # could also be three different projections (=> num_embed * 3)
        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.linear = nn.Linear(embed_size, embed_size)

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
        y = rearrange(y, 'b h t d -> b t (h d)')

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

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # both attn and mlp already have residual dropout
        self.attn = MSMAttention(config.EMBED_SIZE, config.NUM_HEADS)
        self.mlp = FeedForward(config.EMBED_SIZE)
        self.lnorm1 = LayerNorm(config.EMBED_SIZE)
        self.lnorm2 = LayerNorm(config.EMBED_SIZE)
    
    def forward(self, x):
        # using pre-LN https://sushant-kumar.com/blog/normalization-in-transformer-based-llms
        a = self.attn(self.lnorm1(x))
        # residual connection
        y = x + a

        # pre-LN + feedforward
        f = self.mlp(self.lnorm2(y))
        # residual connection
        out = y + f

        return out

class GPT2(nn.Module):
    # https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    # https://medium.com/@vipul.koti333/from-theory-to-code-step-by-step-implementation-and-code-breakdown-of-gpt-2-model-7bde8d5cecd

    def __init__(self, VOCAB_SIZE):
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, config.EMBED_SIZE)
        # (Max Sequence Length -> Embed Size)
        self.pos_embed = nn.Embedding(config.BLOCK_SIZE, config.EMBED_SIZE)
        self.embed_dropout = nn.Dropout(config.RESID_DROP)

        # Repeating transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(config.NUM_LAYERS)])

        self.lnorm = LayerNorm(config.EMBED_SIZE)
        # Final projection to vocab size for word predictions
        self.lm_head = nn.Linear(config.EMBED_SIZE, VOCAB_SIZE)
    
    def forward(self, idx):
        _, T = idx.size()
        # token embedding
        tok_emb = self.token_embed(idx) 

        device = idx.device
        # creates [0, 1, 2, ..., T-1] - used for position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.pos_embed(pos)

        # add position to token embedding and apply dropout
        x = self.embed_dropout(tok_emb + pos_emb)

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.lnorm(x)

        # get next word logits
        logits = self.lm_head(x)
        return logits
    
    # PyTorch will not track gradients for generation
    @torch.no_grad() 
    def generate(self, idx, max_new_tokens, eot_token_id=50256):
        for _ in range(max_new_tokens):
            # crop context to BLOCK_SIZE -- eg. context window
            idx_cond = idx[:, -config.BLOCK_SIZE:]
            
            logits = self(idx_cond)
            
            # multiple predictions get made, we only care about the last new -- eg. the new word
            # fixed by KV-caching
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # stop if we hit <|endoftext|>
            if idx_next.item() == eot_token_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx








    

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        
        self.c_proj.intial_scale = 1 # As the residual need a layer normalizations
        
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self,x):
        B, T, C = x.size() # Batch, Seq_len , n_embed
        
        qkv = self.c_attn(x) # Batch, Seq_len , 3*n_embed
        q, k, v = qkv.split(self.n_embed, dim=-1)
        
        # change q k v to the multi head attention things
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        
        
        # att_score = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att_score = att_score.masked_fill(self.bias[:,:,:T,:T] == 0 , float('-inf') )
        # att = F.softmax(att_score,dim=-1)
        # y = att @ v 
        
        # thee above 4 lines is removed to have the flash_attention layers
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        
        # the above y is in form my B, nh, T, dim
        
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y
        
        
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embed*4, config.n_embed)
        self.c_proj.intial_scale = 1
    
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
   

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
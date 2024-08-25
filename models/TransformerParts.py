import torch
import torch.nn as nn
import torch.nn.functional as F
import einx

class AttentionHead(nn.Module):
    def __init__(self, emb_dim, head_size):
        super().__init__()
        self.keyMap = nn.Linear(emb_dim, head_size, bias=False)
        self.queryMap = nn.Linear(emb_dim, head_size, bias=False)
        self.valueDownMap = nn.Linear(emb_dim, head_size, bias=False)
        self.head_size = head_size

    def forward(self, x):
        # (B, C, emb_dim)
        key = self.keyMap(x) # (B, C, head_size)
        query = self.queryMap(x) # (B, C, head_size)
        valueDown = self.valueDownMap(x) # (B, C, head_size)

        # Self-attention, pairwise dot-prod all queries to all keys then softmax
        attentionDotProd = einx.dot("b q [d], b k [d] -> b q k", query, key) # (B, C, C)
        softMaxPerQuery = einx.softmax("b q [k]", attentionDotProd / (self.head_size ** 0.5))
        # Take the weighted sum of the valueDown heads for each channel, where each channel corresponds to 
        valueDownSum = einx.dot("b q [k], b [k] h -> b q h", softMaxPerQuery, valueDown) # (B C head_size)


        return valueDownSum

class MultiHeadedAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.head_size = emb_dim // num_heads
        self.attention_heads = nn.ModuleList([AttentionHead(emb_dim, self.head_size) for _ in range(num_heads)])
        self.output = nn.Linear(emb_dim, emb_dim, bias=False)
        
    
    def forward(self, x):
        valueArray = []
        for head in self.attention_heads:
            value = head(x)
            valueArray.append(value)
        value = torch.concatenate(valueArray, dim=2)
        output = self.output(value)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadedAttention(emb_dim, num_heads)
        self.feedforward_nonlinear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
    def forward(self, x):
        attentionOutput = self.attention(x)
        x = self.norm1(x + attentionOutput)
        x = self.norm2(x + self.feedforward_nonlinear(x))
        return x
import torch
import torch.nn as nn

from model.Attention import *

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.attention_head = AttentionHead(input_dim, hidden_dim, num_heads, SelfAttention)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        xp = self.attention_head(x)
        x = self.layer_norm_1(xp + x)
        
        xp = self.fc(x)
        return self.layer_norm_2(xp + x)
    

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.self_attention_head = AttentionHead(input_dim, hidden_dim, num_heads, SelfAttention)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)

        self.cross_attention_head = AttentionHead(input_dim, hidden_dim, num_heads, CrossAttention)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm_3 = nn.LayerNorm(hidden_dim)

    def forward(self, x1, x2):
        xp = self.self_attention_head(x1)
        x = self.layer_norm_1(xp + x1)

        xp = self.cross_attention_head((x, x2))
        x = self.layer_norm_2(xp + x)

        xp = self.fc(x)
        return self.layer_norm_3(xp + x)
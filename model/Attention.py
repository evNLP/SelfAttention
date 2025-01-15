import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, inpunt_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.inpunt_dim = inpunt_dim
        self.hidden_dim = hidden_dim
        self.Q = nn.Linear(inpunt_dim, hidden_dim, bias=False)
        self.K = nn.Linear(inpunt_dim, hidden_dim, bias=False)
        self.V = nn.Linear(inpunt_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        xq = self.Q(x)
        xk = self.K(x)
        xv = self.V(x)

        scores = torch.bmm(xq, xk.transpose(1, 2))
        scores = scores / (self.hidden_dim ** 0.5)
        mask = self._mask(scores).to(x.device)
        scores = scores + mask

        scores = self.softmax(scores)
        attention = torch.bmm(scores, xv)
        return scores, attention

    def _mask(self, x):
        mask = torch.tril(torch.ones(x.size(1), x.size(1)), diagonal=0)
        mask[mask == 0] = float('-inf')
        mask[mask == 1] = 0
        mask = mask.repeat(x.size(0), 1, 1)
        return mask
    

class CrossAttention(nn.Module):
    def __init__(self, inpunt_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.inpunt_dim = inpunt_dim
        self.hidden_dim = hidden_dim
        self.Q = nn.Linear(inpunt_dim, hidden_dim, bias=False)
        self.K = nn.Linear(inpunt_dim, hidden_dim, bias=False)
        self.V = nn.Linear(inpunt_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        xq = self.Q(x1)
        xk = self.K(x2)
        xv = self.V(x2)

        scores = torch.bmm(xq, xk.transpose(1, 2))
        scores = scores / (self.hidden_dim ** 0.5)
        #mask = self._mask(scores).to(x1.device)
        #scores = scores + mask

        scores = self.softmax(scores)
        attention = torch.bmm(scores, xv)
        return scores, attention

    def _mask(self, x):
        mask = torch.tril(x[0,:,:], diagonal=0)
        mask[mask == 0] = float('-inf')
        mask[mask == 1] = 0
        mask = mask.repeat(x.size(0), 1, 1)
        return mask
    

class AttentionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, Attention):
        super(AttentionHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.attention_heads = nn.ModuleList([Attention(input_dim, hidden_dim) for _ in range(num_heads)])
        self.fc = nn.Linear(self.num_heads * hidden_dim, hidden_dim)

    def forward(self, x):
        if isinstance(x, tuple):
            x1, x2 = x
            attention_heads = [attention_head(x1, x2) for attention_head in self.attention_heads]
            attention_heads = torch.cat([attention_head[1] for attention_head in attention_heads], dim=-1)
            x = self.fc(attention_heads)
            
        else:
            attention_heads = [attention_head(x) for attention_head in self.attention_heads]
            attention_heads = torch.cat([attention_head[1] for attention_head in attention_heads], dim=-1)
            x = self.fc(attention_heads)
            
        return x
import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, n_embed, head_dim, block_size):
        super().__init__()
        self.k_proj = nn.Linear(n_embed, head_dim)
        self.q_proj = nn.Linear(n_embed, head_dim)
        self.v_proj = nn.Linear(n_embed, head_dim)
        self.head_dim = head_dim

        tril = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('tril', tril)

    def forward(self, x):
        _, T, _ = x.shape
        # (B, T, n_embed)
        k = self.k_proj(x) # (B, T, head_dim)
        q = self.q_proj(x)
        v = self.v_proj(x)

        wei = q @ k.transpose(-2, -1) * self.head_dim**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        y = wei @ v # (B, T, head_dim)
        return y

class PicoGPT(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)
        self.attn = Head(n_embed, n_embed, block_size)
        self.attn_proj = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size
    
    def forward(self, x, target=None):
        # x, target: (B, T)
        _, T = x.shape
        tok_emb = self.token_embedding(x) # (B, T, n_embed)
        pos_emb = self.pos_embedding(torch.arange(T, device=x.device))
        res = tok_emb + pos_emb
        y = self.attn(res) # (B, T, head_dim)
        y = self.attn_proj(y) # (B, T, n_embed)
        y = F.relu(y + res)
        y = self.proj(y) # (B, T, vocab_size)

        if target is None:
            loss = None
        else:
            B, T, C = y.shape
            loss = F.cross_entropy(y.view(B*T, C), target.view(-1))
        return y, loss
    
    def generate(self, x, max_new_tokens):
        # x: (B, T)
        for _ in range(max_new_tokens):
            input = x[:, -self.block_size:]
            logits, _ = self(input) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            x = torch.cat([x, next_token], dim=-1)
        return x
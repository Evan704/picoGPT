import torch
import torch.nn as nn
from torch.nn import functional as F

dropout = 0.2

class Head(nn.Module):
    def __init__(self, n_embed, head_dim, block_size):
        super().__init__()
        self.k_proj = nn.Linear(n_embed, head_dim)
        self.q_proj = nn.Linear(n_embed, head_dim)
        self.v_proj = nn.Linear(n_embed, head_dim)
        self.head_dim = head_dim

        tril = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('tril', tril)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        # (B, T, n_embed)
        k = self.k_proj(x) # (B, T, head_dim)
        q = self.q_proj(x)
        v = self.v_proj(x)

        wei = q @ k.transpose(-2, -1) * self.head_dim**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        y = wei @ v # (B, T, head_dim)
        return y

class MultiHeadAttn(nn.Module):
    def __init__(self, num_head, n_embed, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, n_embed // num_head, block_size) for _ in range(num_head)])
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = torch.cat([h(x) for h in self.heads], dim=-1)
        y = self.dropout(y)
        y = self.out_proj(y)
        return y
    
class FFN(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        hidden_dim = 4*n_embed
        self.net = nn.Sequential(
            nn.Linear(n_embed, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, num_head, n_embed, block_size):
        super().__init__()
        self.attn = MultiHeadAttn(num_head, n_embed, block_size)
        self.ffn = FFN(n_embed)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.norm1(x)
        y = self.attn(x) + x
        y = self.norm2(y)
        y = self.ffn(y) + y
        return y

class PicoGPT(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)
        num_head = 6
        num_block = 6
        self.transformer = nn.Sequential(
            *[TransformerBlock(num_head, n_embed, block_size) for _ in range(num_block)],
            nn.LayerNorm(n_embed)
        )
        self.proj = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size
    
    def forward(self, x, target=None):
        # x, target: (B, T)
        _, T = x.shape
        tok_emb = self.token_embedding(x) # (B, T, n_embed)
        pos_emb = self.pos_embedding(torch.arange(T, device=x.device))
        y = tok_emb + pos_emb
        y = self.transformer(y) # (B, T, n_embed)
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
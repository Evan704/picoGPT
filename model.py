import torch
import torch.nn as nn
from torch.nn import functional as F

class PicoGPT(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)
        self.proj = nn.Linear(n_embed, vocab_size)
        tril = torch.tril(torch.ones(block_size, block_size))
        wei = torch.zeros(block_size, block_size)
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.register_buffer('wei', wei)
        self.block_size = block_size
    
    def forward(self, x, target=None):
        # x, target: (B, T)
        _, T = x.shape
        tok_emb = self.token_embedding(x) # (B, T, n_embed)
        pos_emb = self.pos_embedding(torch.arange(T, device=x.device))
        cur = tok_emb + pos_emb
        y = self.wei[:T, :T] @ cur
        y = F.relu(y + cur)
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
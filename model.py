import torch
import torch.nn as nn
from torch.nn import functional as F

class PicoGPT(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.proj = nn.Linear(n_embed, vocab_size)
    
    def forward(self, x, target=None):
        # x, target: (B, T)
        y = self.token_embedding(x) # (B, T, n_embed)
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
            logits, _ = self(x) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            x = torch.cat([x, next_token], dim=-1)
        return x
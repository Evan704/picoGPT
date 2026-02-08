import numpy as np
import os
from tqdm import trange, tqdm

job_id = os.environ.get('SLURM_JOB_ID')

data_dir = 'datasets/TinyStories'
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
print("Data loaded")

# print(f"Length: {len(text)}")
# print(text[:200])

import tiktoken

enc = tiktoken.get_encoding("gpt2")

encode = lambda s: enc.encode_ordinary(s)
decode = lambda l: enc.decode(l)

vocab_size = 50304
# print(decode(encode('Hello World!')))

import torch
torch.manual_seed(42)

device = 'cuda'

block_size = 256
batch_size = 64

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

from model import PicoGPT
n_embed = 384
model = PicoGPT(vocab_size, n_embed, block_size)
model = model.to(device)

def sample():
    input = torch.tensor([[50256]], dtype=torch.long, device=device)
    print(decode(model.generate(input, max_new_tokens=300)[0].tolist()))

lr = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

@torch.no_grad()
def estimate_loss():
    model.eval()
    esti_iter = 50
    loss = 0
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(esti_iter)
        for k in range(esti_iter):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

print("Start to train...")
train_iter = 20000
eval_interval = 1000
trigger_times = 0
max_trigger_times = 5
best_val_loss = float('inf')

pbar = trange(train_iter, desc="Training", ncols=100)
for iter in pbar:
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    pbar.set_postfix(loss=loss)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        tqdm.write(f"Iter {iter}:")
        losses = estimate_loss()
        cur_val_loss = losses['val']
        tqdm.write(f"train loss: {losses['train']}, val loss: {cur_val_loss}")

        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), f'ckpt/best_model_{job_id}.pth')
            tqdm.write("Best model saved!")
        else:
            trigger_times += 1
            tqdm.write(f"No improvement. Early stopping counter: {trigger_times}")
            if trigger_times >= max_trigger_times:
                tqdm.write(f"Early stopping at {iter}")
                break

sample()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")
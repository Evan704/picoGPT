with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print(f"Length: {len(text)}")
# print(text[:200])

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# print(decode(encode('Hello World!')))

import torch

device = 'cuda'
data = torch.tensor(encode(text), dtype=torch.long, device=device)

# split
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]
# print(data[:200])
print("Data loaded")

block_size = 8
batch_size = 32

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

from model import PicoGPT
n_embed = 32
model = PicoGPT(vocab_size, n_embed, block_size)
model = model.to(device)

def sample():
    input = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(input, max_new_tokens=300)[0].tolist()))

lr = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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
    print(f"Loss in train set: {out['train']}, val set: {out['val']}")
    model.train()

print("Start to train...")
train_iter = 20000
eval_interval = 2000
for iter in range(train_iter):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        print(f"Iter {iter}:")
        estimate_loss()

sample()
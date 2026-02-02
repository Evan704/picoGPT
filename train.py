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
data = torch.tensor(encode(text), dtype=torch.long)
# print(data[:200])

# split
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

block_size = 8
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

from model import PicoGPT
n_embed = 32
model = PicoGPT(vocab_size, n_embed)
# print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
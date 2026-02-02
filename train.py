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
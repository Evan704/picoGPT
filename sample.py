import torch
from model import PicoGPT
import tiktoken

enc = tiktoken.get_encoding("gpt2")
decode = lambda l: enc.decode(l)

vocab_size = 50304
n_embed = 384
block_size = 256
model = PicoGPT(vocab_size, n_embed, block_size)

state_dict = torch.load('ckpt/best_model.pth', map_location='cuda')

model.load_state_dict(state_dict)
model.to(device='cuda')
model.eval()

input = torch.tensor([[50256]], dtype=torch.long, device='cuda')
print(decode(model.generate(input, max_new_tokens=300)[0].tolist()))
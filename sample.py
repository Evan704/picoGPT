import torch
from model import PicoGPT
import tiktoken
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--ckpt', default='best_model.pth')
parser.add_argument('-t', '--temperature', default=0.7, type=float)
parser.add_argument('-k', '--topk', default=50, type=int)
args = parser.parse_args()

enc = tiktoken.get_encoding("gpt2")
decode = lambda l: enc.decode(l)

vocab_size = 50304
n_embed = 384
block_size = 256
num_head = 6
num_block = 6
model = PicoGPT(vocab_size, n_embed, block_size, num_head, num_block)

state_dict = torch.load(f'ckpt/{args.ckpt}', map_location='cuda')
print("State dict loaded.")

model.load_state_dict(state_dict)
model.to(device='cuda')
model.eval()

input = torch.tensor([[50256]], dtype=torch.long, device='cuda')
print(decode(
    model.generate(
        input, max_new_tokens=300,
        temperature=args.temperature,
        top_k=args.topk
    )[0].tolist()
))
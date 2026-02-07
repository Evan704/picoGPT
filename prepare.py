import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

num_proc = 8
dtype = np.uint16

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("Loading dataset...")
split_dataset = load_dataset("text", data_files={
    'train': "datasets/TinyStories/TinyStories-train.txt",
    'val': "datasets/TinyStories/TinyStories-valid.txt"
})
print("Dataset loaded.")

enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token) # 添加结束符 <|endoftext|>
    return {'ids': ids, 'len': len(ids)}

tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

for split, set in tokenized.items():
    arr_len = np.sum(set['len'], dtype=np.uint64)
    filename = os.path.join('datasets/TinyStories/', f'{split}.bin')
    print(f"Writing {filename}")
    
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    total_batches = 1024
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
        batch = set.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx:idx+len(arr_batch)] = arr_batch
        idx += len(arr_batch)
        
    arr.flush()

print("Done!")
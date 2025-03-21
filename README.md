# AnnAttention
ANN Attention uses the HNSW algorithm to insert and query vectors organized in blocks when querying queries, achieving N * logN complexity and paving the way for LLM contexts of one billion tokens or even longer.


## Features 😎
- Use **faiss.IndexHNSWFlat** accelerates vector insertion and retrieval, reducing the computational complexity of long sequences.
- By using **sliding window** and **block attention** mode, we avoid manipulating at every token, and instead only insert and retrieve block. This batch processing reduces the frequency of operations and increases processing efficiency.


## Complexity
| Mechanism              | Computational Complexity | Memory Complexity |
|------------------------|--------------|----------|
| **Standard Attention** | O(n^2)       | O(n^2)   |
| **ANN Attention**      | O(n log n)   | O(n)     |


## Quickstart 🚀
- Download repository
```
git clone https://github.com/StudyExchange/AnnAttention.git
```

- Install pkgs
```
cd AnnAttention
pip install -r requirements.txt
```

- Download datasets
```
modelscope download --dataset gongjy/minimind_dataset  pretrain_hq.jsonl    --local_dir ./dataset
modelscope download --dataset gongjy/minimind_dataset sft_mini_512.jsonl    --local_dir ./dataset
```

- Run all
```
bash run_all.sh
```


## Training time
Env: Intel i7-14700K, 128G, 4090D(24G)
- Minimind(FlashAttention), training train_pretrain 2 epochs and train_full_sft 2 epochs takes about 4 hours.
- AnnAttention, Training train_pretrain 2 epochs and train_full_sft 2 epochs takes about 4 days. Faiss calculates in CPU, it's too slow.


## Core code
faiss_block_attention.py
```python
import torch
import torch.nn.functional as F
import math
from typing import Optional
import numpy as np
import faiss
import time
from model.sliding_window import sliding_window_with_padding
from model.convert_indices import block_indices_to_item_indices

def select_block(xq, xk, xv, index, top_n, block_size, training):
    seq_len4q, head_dim = xq.shape
    seq_len4k, head_dim = xk.shape
    seq_len4v, head_dim = xv.shape
    nq = xq.cpu().detach().numpy()
    nk = xk.cpu().detach().numpy()

    selected_xk = []
    selected_xv = []
    for i in range(0, seq_len4k, block_size):
        bq = nq[i:i+block_size]
        distances, block_indices = index.search(bq, top_n)
        bk = nk[i:i+block_size]
        block_indices = np.sort(block_indices, axis=1)
        indices = block_indices_to_item_indices(block_indices, block_size, top_n)
        indices = np.where(indices < 0, 0, indices)
        indices_tensor = torch.from_numpy(indices).long()
        selected_xk.append(xk[indices_tensor, :])
        selected_xv.append(xv[indices_tensor, :])
        if bk.shape[0] == block_size and (training or (not training and i == seq_len4k - block_size)):
            bk_mean = np.mean(bk, keepdims=True, axis=0)
            index.add(bk_mean)
    selected_xk_tensor = torch.concatenate(selected_xk, dim=0)
    selected_xv_tensor = torch.concatenate(selected_xv, dim=0)
    return selected_xk_tensor, selected_xv_tensor


def faiss_block_attention(hq, hk, hv, dropout_p: float = 0.0, 
                           window_size: int = 32, block_size: int = 32,
                           top_n: int = 8, training: bool=False, vec_indices: list = []):
    t0 = time.time()
    debug = False
    batch_size, n_head, seq_len4q, head_dim = hq.shape
    batch_size, n_head, seq_len4k, head_dim = hk.shape
    batch_size, n_head, seq_len4v, head_dim = hk.shape
    embed_dim = n_head * head_dim
    
    xq = hq.transpose(1, 2).reshape(batch_size, seq_len4q, embed_dim)
    xk = hk.transpose(1, 2).reshape(batch_size, seq_len4k, embed_dim)
    xv = hv.transpose(1, 2).reshape(batch_size, seq_len4v, embed_dim)

    t1 = time.time()
    xk_windows = sliding_window_with_padding(xk, window_size)
    xv_windows = sliding_window_with_padding(xv, window_size)
    t_diff = time.time() - t1
    if debug:
        print(f'faiss_block_attention --> windows: {t_diff:.4f} s')

    batch_xk = []
    batch_xv = []
    t2 = time.time()
    for batch_i in range(batch_size):
        t00 = time.time()
        index = vec_indices[batch_i]
        selected_xk, selected_xv = select_block(xq[batch_i], xk[batch_i], xv[batch_i], index, top_n, block_size, training)
        batch_xk.append(selected_xk)
        batch_xv.append(selected_xv)
        t_diff = time.time() - t00
        if debug:
            print(f'faiss_block_attention --> batch {batch_i}: {t_diff:.4f} s')
    t_diff = time.time() - t2
    if debug:
        print(f'faiss_block_attention --> batch all: {t_diff:.4f} s')

    t3 = time.time()
    selected_xk_tensor = torch.stack(batch_xk, dim=0)
    selected_xv_tensor = torch.stack(batch_xv, dim=0)
    if xk_windows.shape[1] != selected_xk_tensor.shape[1]:
        xk_windows = xk_windows[:, -1:, :, :]
        xv_windows = xv_windows[:, -1:, :, :]
    batch_xk_tensor = torch.concatenate([xk_windows, selected_xk_tensor], dim=2)
    batch_xv_tensor = torch.concatenate([xv_windows, selected_xv_tensor], dim=2)
    t_diff = time.time() - t3
    if debug:
        print(f'faiss_block_attention --> batch_xk_tensor, batch_xv_tensor: {t_diff:.4f} s')

    t4 = time.time()
    scores = torch.einsum('bqd,bqwd->bqw', xq, batch_xk_tensor) / math.sqrt(embed_dim)
    scores = F.softmax(scores, dim=-1)
    scores = F.dropout(scores, dropout_p)

    output = torch.einsum('bqw,bqwd->bqd', scores, batch_xv_tensor)
    t_diff = time.time() - t4
    if debug:
        print(f'faiss_block_attention --> scores, output: {t_diff:.4f} s')

    output = output.view(batch_size, seq_len4q, n_head, head_dim).transpose(1, 2)
    t_diff = time.time() - t0
    if debug:
        print(f'faiss_block_attention Function internal timing:{t_diff:.4f} s')
    return output

```


## Thanks
- [minimind](https://github.com/jingyaogong/minimind), peripheral code copy from this project.
- [nanoGPT](https://github.com/karpathy/nanoGPT), start learning from this project.


## License
This repository is licensed under the Apache-2.0 License.

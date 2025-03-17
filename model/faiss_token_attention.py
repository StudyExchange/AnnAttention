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
        if training:
            bq = nq[i:i+block_size]
            distances, indices = index.search(bq, top_n)
            bk = nk[i:i+block_size]
            if bk.shape[0] == block_size:
                # bk_mean = np.mean(bk, keepdims=True, axis=0)
                index.add(bk)
            else:
                a = 10
            # 取数据
            # if i == 0:
            #     pad_len = min(block_size, seq_len4q)
            #     indices_tensor = torch.zeros(pad_len, top_n).long()
            # else:
            indices = np.where(indices == -1, 0, indices)
            indices = np.sort(indices, axis=1)
            indices_tensor = torch.from_numpy(indices).long()
            selected_xk.append(xk[indices_tensor, :])
            selected_xv.append(xv[indices_tensor, :])
        else:
            a = 10
            if i >= seq_len4k - block_size:
                bq = nq[-1:]
                distances, indices = index.search(bq, top_n)
                # 取数据
                # if i == 0:
                #     pad_len = min(block_size, seq_len4q)
                #     indices_tensor = torch.zeros(pad_len, top_n).long()
                # else:
                indices = np.where(indices == -1, 0, indices)
                indices = np.sort(indices, axis=1)
                indices_tensor = torch.from_numpy(indices).long()
                selected_xk.append(xk[indices_tensor, :])
                selected_xv.append(xv[indices_tensor, :])
            bk = nk[i:i+block_size]
            if seq_len4q == 1:
                if i >= seq_len4k - block_size:
                    if bk.shape[0] == block_size:
                        index.add(bk)
                    else:
                        a = 10
                else:
                    a = 10
            else:
                if bk.shape[0] == block_size:
                    index.add(bk)

    selected_xk_tensor = torch.concatenate(selected_xk, dim=0)
    selected_xv_tensor = torch.concatenate(selected_xv, dim=0)
    # if not training and seq_len4q == 1:
    #     selected_xk_tensor = selected_xk_tensor[-1:, :, :]
    #     selected_xv_tensor = selected_xv_tensor[-1:, :, :]
    if not selected_xv_tensor.shape[0]:
        a = 10
    return selected_xk_tensor, selected_xv_tensor


def faiss_token_attention(hq, hk, hv, dropout_p: float = 0.0, 
                           window_size: int = 32, block_size: int = 32,
                           top_n: int = 8, training: bool=False, vec_indices: list = []):
    t0 = time.time()
    debug = False
    # 获取基本张量尺寸
    batch_size, n_head, seq_len4q, head_dim = hq.shape
    batch_size, n_head, seq_len4k, head_dim = hk.shape
    batch_size, n_head, seq_len4v, head_dim = hk.shape
    embed_dim = n_head * head_dim
    
    # 将 hq, hk, hv 变换为形状 [batch_size, seq_len, embed_dim] 便于后续处理
    xq = hq.transpose(1, 2).reshape(batch_size, seq_len4q, embed_dim)
    xk = hk.transpose(1, 2).reshape(batch_size, seq_len4k, embed_dim)
    xv = hv.transpose(1, 2).reshape(batch_size, seq_len4v, embed_dim)

    # slidding_window
    t1 = time.time()
    xk_windows = sliding_window_with_padding(xk, window_size)
    xv_windows = sliding_window_with_padding(xv, window_size)
    t_diff = time.time() - t1
    if debug:
        print(f'faiss_token_attention --> windows: {t_diff:.4f} s')

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
            print(f'faiss_token_attention --> batch {batch_i}: {t_diff:.4f} s')
    t_diff = time.time() - t2
    if debug:
        print(f'faiss_token_attention --> batch all: {t_diff:.4f} s')

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
        print(f'faiss_token_attention --> batch_xk_tensor, batch_xv_tensor: {t_diff:.4f} s')

    t4 = time.time()
    scores = torch.einsum('bqd,bqwd->bqw', xq, batch_xk_tensor) / math.sqrt(embed_dim)
    scores = F.softmax(scores, dim=-1)
    scores = F.dropout(scores, dropout_p)

    output = torch.einsum('bqw,bqwd->bqd', scores, batch_xv_tensor)
    t_diff = time.time() - t4
    if debug:
        print(f'faiss_token_attention --> scores, output: {t_diff:.4f} s')

    output = output.view(batch_size, seq_len4q, n_head, head_dim).transpose(1, 2)
    t_diff = time.time() - t0
    if debug:
        print(f'faiss_token_attention Function internal timing:{t_diff:.4f} s')
    return output

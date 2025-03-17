import torch
import torch.nn.functional as F

def sliding_window_with_padding(x, window_size):
    batch_size, seq_len, embed_dim = x.shape
    padding_size = window_size - 1
    # 从 seq_len 维度取出第一个元素，并扩展后作为左侧 padding
    left_pad = x[:, 0:1, :].expand(batch_size, padding_size, embed_dim)
    x_padded = torch.cat([left_pad, x], dim=1)
    new_shape = (batch_size, seq_len, window_size, embed_dim)
    new_strides = (x_padded.stride(0), x_padded.stride(1), x_padded.stride(1), x_padded.stride(2))
    windows = x_padded.as_strided(new_shape, new_strides)
    return windows

if __name__ == "__main__":
    import time

    t0 = time.time()
    batch_size = 1
    seq_len = 7
    embed_dim = 1
    x = torch.arange(1, seq_len + 1).reshape(1, seq_len, 1).expand(batch_size, seq_len, embed_dim)
    window_size = 5
    windows = sliding_window_with_padding(x, window_size)
    print(windows.shape)
    # print(windows)
    t_diff = time.time() - t0
    print(f"Time: {t_diff:.4f} s")

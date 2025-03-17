import numpy as np

def block_indices_to_item_indices(block_indices, block_size, top_n):
    """
    将块索引转换为项索引。

    参数：
        block_indices (numpy.ndarray): 形状为 (num_blocks, 4) 的块索引。
        block_size (int): 每个块的大小。

    返回：
        numpy.ndarray: 形状为 (num_blocks, 4 * block_size) 的项索引。
    """

    num_blocks = block_indices.shape[0]

    # 创建一个形状为 (block_size,) 的数组，包含 0 到 block_size-1 的值
    offsets = np.arange(block_size)

    # 将 offsets 扩展为 (1, block_size) 的形状，以便广播
    offsets = offsets.reshape(1, block_size)

    # 将 block_indices 扩展为 (num_blocks, 4, 1) 的形状，以便广播
    block_indices = block_indices.reshape(num_blocks, top_n, 1)

    # 使用广播添加偏移量
    item_indices = block_indices * block_size + offsets

    # 将形状重塑为 (num_blocks, 4 * block_size)
    item_indices = item_indices.reshape(num_blocks, top_n * block_size)

    return item_indices

if __name__ == "__main__":
    # 示例用法
    batch_size = 2
    top_n = 3
    # block_indices = np.arange(0, batch_size * top_n).reshape(batch_size, top_n)  # 创建一些随机的块索引
    block_indices = np.ones((batch_size, top_n))  # 创建一些随机的块索引
    block_size = 6

    item_indices = block_indices_to_item_indices(block_indices, block_size, top_n)

    print("block_indices 的形状:", block_indices.shape)
    print("item_indices 的形状:", item_indices.shape)

import torch

def allocate_kv_blocks(
    seq_lengths: list,
    block_size: int,
    free_block_ids: list,
) -> tuple:
    """
    Returns (block_table, blocks_used, remaining_free_blocks), integer tensors.
    """
    #先看每seq 需要多少block 
    blocks_need = []
    for length in seq_lengths:
        if length % block_size:
            blocks_need.append(length//block_size+1)
        else:
            blocks_need.append(length//block_size)

    total_need = sum(blocks_need)

    if total_need>len(free_block_ids):
        raise RuntimeError(f"Insufficient free blocks: need {total_need}, have {len(free_block_ids)}")

    max_block =  max(blocks_need)
    block_table = torch.full((len(seq_lengths),max_block),-1,dtype = torch.int64)

    cursor = 0
    # 每个序列发多少block
    for i, n in enumerate(blocks_need):
        assigned = free_block_ids[cursor:cursor+n]
        cursor+=n
        if n>0:
            block_table[i,:n] = torch.tensor(assigned,dtype = torch.int64)

    remaining_free_blocks = torch.tensor(free_block_ids[cursor:],dtype = torch.int64)
    blocks_used = torch.tensor(blocks_need, dtype=torch.int64)

    return block_table, blocks_used, remaining_free_blocks
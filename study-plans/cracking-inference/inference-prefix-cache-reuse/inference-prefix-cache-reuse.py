import torch

def match_prefix_cache(
    request_token_ids: list,
    cached_token_blocks: list,
    cached_physical_block_ids: list,
    block_size: int,
) -> tuple:
    """
    Returns (matched_token_count, reusable_physical_block_ids), scalar and 1D integer tensors.
    """
    num_blocks = len(request_token_ids)// block_size
    request_blocks =[]
    for i in range(num_blocks):
        request_blocks.append(request_token_ids[i*block_size : i*block_size+block_size])

    max_block = 0
    max_block_id = 0
    for i, n in enumerate(cached_token_blocks):
        m = 0
        for j in range(0,min(len(n),len(request_blocks))):
            if request_blocks[j]==n[j]:
                # print(request_blocks[j])
                # print(n)
                m = m+1
            else:
                break
                
        if m > max_block:
            max_block_id = i
            max_block = m 

        # print(max_block_id)
        # print(max_block)

    matched_token_count = torch.tensor(max_block * block_size,dtype = torch.int64)
    reusable_physical_block_ids = torch.tensor(cached_physical_block_ids[max_block_id][:max_block],dtype = torch.int64)
    print(matched_token_count,reusable_physical_block_ids)
    return matched_token_count,reusable_physical_block_ids
        
                
    
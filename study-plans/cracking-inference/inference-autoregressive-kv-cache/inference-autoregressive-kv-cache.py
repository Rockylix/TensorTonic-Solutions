import torch

def cached_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple:
    """
    Returns (outputs, key_cache, value_cache), float32 tensors in sequence order.
    """
    batch, seq , D = query.shape

    outputs = []
    key_cache = None
    value_cache = None

    for t in range(seq):
        single_query = query[:,t:t+1,:]

        single_key = key[:,t:t+1,:]
        single_value =value[:,t:t+1,:]

        if key_cache is None:
            key_cache  = single_key
            value_cache = single_value
        else:
            key_cache = torch.cat([key_cache,single_key],dim=1)
            value_cache = torch.cat([value_cache,single_value],dim=1)

        scores = single_query@key_cache.transpose(-1,-2) / (D**0.5)
        weight = torch.softmax(scores,dim=-1)
        attn = weight@value_cache
        outputs.append(attn)

    print(outputs)
    # 把列表转换为张量 torch.cat的第一个参数是存储张量的列表
    outputs = torch.cat(outputs,dim=1)
    return outputs,key_cache,value_cache
    
        

        
        

        

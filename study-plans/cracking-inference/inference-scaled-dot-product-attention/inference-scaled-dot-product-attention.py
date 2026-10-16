import torch

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask = None,
) -> torch.Tensor:
    """
    Returns a float32 attention tensor with shape (batch, query length, value width).
    """
    _, _, dk = query.shape
    s = torch.matmul(query,key.transpose(1,2)) / (dk**0.5)
    # print(key.transpose(1,2).shape)
    # print(query.shape)
    # 做mask
    if mask is not None:
        s = s.masked_fill(mask,float("-inf")) 

    # softmax weight[batch, seq_q ,seq_kv]
    weight = torch.softmax(s,dim=-1)
    #v = [batch, seq_kv, v_dim]
    return torch.matmul(weight,value)
    
    

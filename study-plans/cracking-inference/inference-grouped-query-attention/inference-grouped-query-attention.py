import torch

def grouped_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor with the same shape as hidden_states.
    """
    batch,seq,D = hidden_states.shape
    query = hidden_states@w_q
    key = hidden_states@w_k
    value = hidden_states@w_v

    d = D // num_query_heads
    if num_query_heads ==0 or num_kv_heads ==0:
        raise ValueError("invalid")
    elif num_query_heads %num_kv_heads :
        raise ValueError("invalid")
        
    group = num_query_heads // num_kv_heads

    query = query.reshape(batch,seq,num_query_heads,d).transpose(1,2)
    key = key.reshape(batch,seq,num_kv_heads,d).transpose(1,2)
    value = value.reshape(batch,seq,num_kv_heads,d).transpose(1,2)

    key = key.repeat_interleave(group,dim=1)
    value = value.repeat_interleave(group,dim=1)

    scores = query@ (key.transpose(-1,-2)) / (d**0.5)

    if causal:
        mask = torch.full_like(scores,True,dtype = bool)
        mask = torch.triu(mask,diagonal=1)
        scores=scores.masked_fill(mask,float("-inf"))

    weight = torch.softmax(scores,dim=-1)
    attn = weight@value
    concat = attn.transpose(1,2).reshape(batch,seq,D)
    return concat@w_o
    
    
    

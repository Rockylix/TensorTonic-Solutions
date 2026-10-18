import torch

def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as hidden_states.
    """
    # print(hidden_states.shape)
    #[B,S,D]
    batch,seq,D = hidden_states.shape
    #[b,s,dq]
    query = torch.matmul(hidden_states,w_q)
    key = torch.matmul(hidden_states,w_k)
    value = torch.matmul(hidden_states,w_v)
    
    #拆分多头
    d = D // num_heads
    #[batch,numshead,seq,d]
    def split_head(x:torch.Tensor) -> torch.Tensor:
        return x.reshape(batch,seq,num_heads,d).transpose(1,2)
        
    query = split_head(query)
    key = split_head(key)
    value = split_head(value)
    # print(query.shape)

    # 注意力权重 [batch,numshead,seq_q,seq_k]
    scores = torch.matmul(query,key.transpose(-1,-2)) / (d**0.5)
    # print(scores.shape)
    if causal is not False:
        mask = torch.full_like(scores,True,dtype=bool)
        mask = torch.triu(mask,diagonal=1)
        print(mask)
        scores = torch.where(mask,torch.full_like(scores,float("-inf")),scores)
        # scores = scores.masked_fill(mask,float("-inf"))

    weight = torch.softmax(scores,axis = -1)
    #[batch,numshead,seq,d]
    attn = torch.matmul(weight,value)
    # concat = H_head.transpose(1,2).reshape(batch,seq,D)
    concat = attn.transpose(1,2).contiguous().view(batch,seq,D)
    return torch.matmul(concat,w_o)

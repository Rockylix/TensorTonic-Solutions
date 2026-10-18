import torch

def multi_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor with the same shape as hidden_states.
    """
    batch,seq,D = hidden_states.shape
    #[batch,seq,D]
    query = hidden_states@w_q
    #[batch,seq,d]
    key = hidden_states@w_k
    value = hidden_states@w_v
    d = D // num_query_heads
    # 划分多query [batch,numheads,seq,d]
    query = query.reshape(batch,seq,num_query_heads,d).transpose(1,2)
    #k 需要对齐维度， unsqueeze 一个head维度
    scores = torch.matmul(query,key.unsqueeze(1).transpose(-1,-2)) / (d**0.5)

    if causal is not False:
        mask = torch.full_like(scores,True,dtype=bool)
        # 代表严格上三角
        mask = torch.triu(mask,diagonal=1)
        scores = scores.masked_fill(mask,float("-inf"))

    weight = torch.softmax(scores,axis = -1)
    attn = torch.matmul(weight,value.unsqueeze(1))

    concat = attn.transpose(1,2).reshape(batch,seq,D)
    return concat@w_o
    
    
    

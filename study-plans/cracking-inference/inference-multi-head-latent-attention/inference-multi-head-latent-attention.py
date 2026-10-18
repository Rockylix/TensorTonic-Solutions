import torch

def multi_head_latent_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_down: torch.Tensor,
    w_up_k: torch.Tensor,
    w_up_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> tuple:
    """
    Returns (output, latent), tensors shaped (batch, seq, model width) and (batch, seq, latent width).
    """
    batch,seq,D = hidden_states.shape
    query = hidden_states@w_q
    C = hidden_states@w_down
    key = C@w_up_k
    value = C@w_up_v

    d = D // num_heads
    def split(x:torch.Tensor) -> torch.Tensor:
        return x.reshape(batch,seq,num_heads,d).transpose(1,2)

    query=split(query)
    key = split(key)
    value = split(value)

    scores = query@key.transpose(-1,-2) / (d**0.5)

    if causal is not False:
        mask = torch.full_like(scores,True,dtype=bool)
        mask = torch.triu(mask,diagonal=1)
        scores = scores.masked_fill(mask,float("-inf"))

    weight = torch.softmax(scores,dim=-1)
    attn = weight@value
    concat = attn.transpose(1,2).reshape(batch,seq,D)
    return (concat@w_o,C)
                
    
    

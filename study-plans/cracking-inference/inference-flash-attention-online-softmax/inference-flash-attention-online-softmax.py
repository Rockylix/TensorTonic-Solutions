import torch

def flash_attention_online_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_block_size: int,
    key_block_size: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor shaped (batch, query length, value width).
    """
    batch,seq_q,d_q = query.shape
    _,seq_kv,d_v = value.shape
    output = torch.zeros((batch,seq_q,d_v),dtype=torch.float)

    for start_q in range(0,seq_q,query_block_size):
        end_q = min(seq_q,start_q+query_block_size)
        query_block = query[:,start_q:end_q,:]
        #实际的块是qb 
        qb = end_q - start_q
        
        # 维护变量
        m_i = torch.full((batch,qb),float("-inf"),dtype=torch.float)
        l_i = torch.zeros((batch,qb),dtype = torch.float)
        acc = torch.zeros((batch,qb,d_v),dtype=torch.float)

        for start_k in range(0,seq_kv,key_block_size):
            end_k = min(seq_kv,start_k+key_block_size)
            key_block = key[:,start_k:end_k,:]
            value_block = value[:,start_k:end_k,:]

            scores = query_block@key_block.transpose(-1,-2) / (d_q**0.5)

            if causal is not False:
                #[query,]
                query_idx = torch.arange(start_q,end_q).unsqueeze(1)
                #[,key]
                key_idx = torch.arange(start_k,end_k).unsqueeze(0)
                mask = key_idx>query_idx
                scores = scores.masked_fill(mask,float("-inf"))

            # m [batch,query_q]
            # score[batch,query_q,query_kv]
            #alpha [batch, query_q]
            m_new = torch.maximum(m_i,scores.max(dim=-1).values)
            
            def safe_exp(x:torch.Tensor,ref:torch.Tensor) -> torch.Tensor:
                diff = x-ref
                return torch.where(torch.isneginf(ref),torch.zeros_like(x),torch.exp(diff))
                
            alpha = safe_exp(m_i,m_new)
            p = safe_exp(scores,m_new.unsqueeze(-1))
            
            acc = acc*alpha.unsqueeze(-1) + p@value_block
            l_i = l_i*alpha + p.sum(dim=-1)
            m_i = m_new

        o_i = acc / l_i.unsqueeze(-1)
        output[:,start_q:end_q,:] = o_i

    return output 
            
            
            
            
    

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
    batch, seq_q, dk = query.shape
    _, seq_k,dv = value.shape
    output = torch.zeros(batch,seq_q,dv,dtype=query.dtype)

    for q_start in range(0,seq_q,query_block_size):
        #q_mask
        q_end = min(q_start+query_block_size,seq_q)
        #取q
        q = query[:,q_start:q_end,:]
        qb = q_end - q_start

        # 维护向量
        m_i = torch.full((batch,qb),float("-inf"),dtype = query.dtype)
        l_i = torch.zeros((batch,qb),dtype = query.dtype)
        acc = torch.zeros((batch,qb,dv),dtype =query.dtype)
        def _safe_exp(x:torch.Tensor,ref:torch.Tensor)->torch.Tensor:
            diff =x-ref;
            return torch.where(torch.isneginf(ref),torch.zeros_like(diff),torch.exp(diff))
        
        for k_start in range(0,seq_k, key_block_size):
            k_end = min(k_start+key_block_size,seq_k)
            #取k v
            k = key[:,k_start:k_end,:]
            v = value[:,k_start:k_end,:]
            kv_blocksize = k_end - k_start

            # s 是 [batch,qb,kv_blocksize]
            s = torch.matmul(q,k.transpose(-2,-1)) /(dk **0.5)
            #s 超出的 部分不用做了，因为torch之间在数据维度上限制了

            ## 处理causal的
            if causal:
                q_idx = torch.arange(q_start,q_end).unsqueeze(1)
                k_idx = torch.arange(k_start,k_end).unsqueeze(0)
                mask = k_idx>q_idx
                s = s.masked_fill(mask,float("-inf"))

            # m_new 是 [batch, qb,]
            # max 返回的是一个元组， 里面又values 和 indices  因此要取出values
            m_new = torch.maximum(m_i,s.max(dim=-1).values)

            alpha = _safe_exp(m_i,m_new)
            #要对齐维度 拓展最后一个维度
            p = _safe_exp(s,m_new.unsqueeze(-1))

            l_i = alpha *l_i + p.sum(dim=-1)
            #acc [batch,qb,dv] alpha 是 [batch, qb]
            acc = acc * alpha.unsqueeze(-1) + torch.matmul(p,v)
            m_i = m_new

        output[:,q_start:q_end,:] = acc / l_i.unsqueeze(-1)

    return output
            
            
            
            
    

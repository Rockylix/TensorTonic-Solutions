import torch

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    uniform_draws: torch.Tensor,
) -> torch.Tensor:
    """
    Returns an int64 tensor of token IDs with shape (batch,).
    """
    bacth, vocab = logits.shape
    #返回索引
    if temperature==0:
         return torch.argmax(logits,dim=-1)
    
    scaled = logits / temperature

    #选topk  sort 返回索引序列
    if top_k>0 and top_k<vocab:
        topk_index = torch.sort(scaled,dim=-1,descending=True,stable = True).indices[...,:top_k]
        mask = torch.zeros_like(scaled,dtype=torch.bool)
        # 参数分别为 dim ， index 位置， 填充内容
        mask.scatter_(-1,topk_index,True)
        scaled = scaled.masked_fill(~mask,float("-inf"))
    #topk 的概率
    pro_b = torch.softmax(scaled,dim=-1)
    if top_p < 1:
        sort_pro, sort_index = torch.sort(pro_b,dim=-1,descending=True,stable=True)
        # 前缀和序列
        cumulative = torch.cumsum(sort_pro,dim=-1)
        # 看看到哪里满足
        keep_sort = (cumulative-sort_pro) < top_p
        #至少保留第一个
        keep_sort[...,0] =True
        mask = torch.zeros_like(sort_pro,dtype=bool)
        #填入sort里面对应的值
        mask.scatter_(-1,sort_index,keep_sort)
        pro_b = pro_b.masked_fill(~mask,0.0)

    final_pro = pro_b/pro_b.sum(dim=-1,keepdim=True)

    #随机数采样
    cumulative = torch.cumsum(final_pro,dim=-1)
    #uniform_draws 是一个数， 要扩展
    draws = uniform_draws.unsqueeze(-1)
    sampled = torch.searchsorted(cumulative,draws,right = True).squeeze(-1)
    return torch.clamp(sampled,max = vocab-1)
        
        
    
        
    
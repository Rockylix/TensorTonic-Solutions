import torch

def verify_speculative_tokens(
    draft_token_ids: torch.Tensor,
    draft_distributions: torch.Tensor,
    target_distributions: torch.Tensor,
    uniform_draws: torch.Tensor,
) -> tuple:
    """
    Returns (accepted_prefix, accepted_count, next_token), 1D, scalar and scalar integer tensors.
    """
    K = draft_token_ids.shape[0]
    accepted_count = 0

    for i in range(K):
        # 草稿模型在 i 位置预测的 token id 
        token = draft_token_ids[i].item() #转换为原生标量 这里就是 int
        p_draft = draft_distributions[i,token].item() # 这是一个[K,vocab] 也就是在 i位置，草稿模型给这个token分布的概率是多少
        p_target = target_distributions[i,token].item()
        print(token,p_draft,p_target)

        #接收概率 target 分配的概率更高就选这个 token
        accept_prob = min(1.0,p_target/p_draft) if p_draft>0 else 0.0 

        # 用随机数来判断  ac_prob 为1 一定接收， 为0 一定拒绝，  为0.6 就是和随机数比，随机接收
        if uniform_draws[i].item() <accept_prob:
            accepted_count+=1
        else:
            break

    # 预测全对， 此时直接拿 target_distributions[K] 从这个分布采样出的 token 叫 bonus token（奖励 token）：target 白送一个，不需要 draft 再猜
    if accepted_count == K:
        next_dist = target_distributions[K]
    else:
        #正残差， draft 给多了 接收机制按pt/pd 比例拒绝掉多余部分，此时残差就为0 ， 
        # draft 给少了， 接受机制全收 ， 但是target 要的那部分缺口 由 残差这里补上  
        # 保证最后的总量为 p_target 
        residual = torch.clamp(
        target_distributions[accepted_count] - draft_distributions[accepted_count], min=0.0
        )
        next_dist = residual / residual.sum()

    # 前缀和
    cumulative  = torch.cumsum(next_dist,dim=-1)
    draw = uniform_draws[K].unsqueeze(0)
    next_token = torch.searchsorted(cumulative,draw,right=True).squeeze(0)
    next_token = torch.clamp(next_token, max=next_dist.shape[0] - 1) 

    accepted_prefix = draft_token_ids[:accepted_count]
    return accepted_prefix, torch.tensor(accepted_count,dtype = torch.int64), next_token
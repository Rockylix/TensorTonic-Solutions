import torch

def sparse_moe_forward(
    token_states: torch.Tensor,
    router_logits: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Returns an output tensor with the same shape as token_states.
    """
    seq, D = token_states.shape
    num_expert = router_logits.shape[-1]
    # 选专家 rounter_logits [seq, expert]
    expert_value, expert_index = torch.sort(router_logits,dim=-1,descending=True,stable=True)
    # print(expert_index,expert_value)
    expert_topk = expert_index[...,:top_k]
    # print(expert_topk) 
    # [seq,expert[...,:topk]]
    rounting_weights = torch.softmax(expert_value[...,:top_k],dim=-1)
    # print(rounting_weights)

    output = torch.zeros_like(token_states)

    for n in range(num_expert): # 这里是遍历每一个专家， 看这个专家有哪些token使用， 然后 取出对应token 进行计算
        # 因为每个token 都有 nums_expert 去选， 因此mask 一定是一个[seq,topk]
        mask = expert_topk == n
        if not mask.any(): # 没有为真的，那就不在任何token的选择专家里面
            continue

        # 找出 mask 里面 不为0的 行下标和 列下标
        token_idx,col_idx = mask.nonzero(as_tuple=True)
        # [token, Dimension]
        x = token_states[token_idx]
        # print(x)

        # 执行算法 w[n] 是第n个专家的权重 为 [D ,F] F 为专家的隐藏维度
        # [token ,F]
        hidden = torch.relu(x@w_in[n])
        # [token ,D]
        expert_out = hidden@w_out[n]

        #加权 rounting_weights 是 [seq,expert[...,:topk]] 取出来之后 [token]
        weights = rounting_weights[token_idx,col_idx].unsqueeze(-1)
        print(weights)
        output.index_add_(0,token_idx,expert_out*weights)

    return output
        
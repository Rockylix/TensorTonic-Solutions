import torch

def route_tokens_to_experts(
    router_logits: torch.Tensor,
    top_k: int,
) -> tuple:
    """
    Returns (expert_indices, routing_weights), tensors shaped (num_tokens, top_k).
    """
    sorted_value,sorted_index = torch.sort(router_logits,dim=-1,descending=True,stable=True)
    expert_index = sorted_index[...,:top_k]
    expert_value = sorted_value[...,:top_k]
    print(expert_index)
    print(expert_value)

    routing_weights = torch.softmax(expert_value,dim=-1)
    return (expert_index,routing_weights)

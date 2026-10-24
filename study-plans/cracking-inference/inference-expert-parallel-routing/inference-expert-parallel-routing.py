import torch

def prepare_expert_parallel_dispatch(
    selected_expert_ids: torch.Tensor,
    expert_to_rank: torch.Tensor,
    routing_weights: torch.Tensor,
    world_size: int,
) -> tuple:
    """
    Returns (dispatch_permutation, per_rank_send_counts, inverse_permutation,
    reordered_expert_ids, reordered_weights): four integer tensors and one floating tensor.
    """
    flat_expert_ids = selected_expert_ids.reshape(-1)
    flat_weights = routing_weights.reshape(-1)
    dest_ranks = expert_to_rank[flat_expert_ids]

    dispatch_permutation = torch.argsort(dest_ranks, stable=True)
    per_rank_send_counts = torch.bincount(dest_ranks, minlength=world_size).to(torch.int64)

    inverse_permutation = torch.empty_like(dispatch_permutation)
    inverse_permutation[dispatch_permutation] = torch.arange(
        dispatch_permutation.shape[0], dtype=torch.int64
    )

    reordered_expert_ids = flat_expert_ids[dispatch_permutation]
    reordered_weights = flat_weights[dispatch_permutation]

    return (dispatch_permutation, per_rank_send_counts, inverse_permutation,
            reordered_expert_ids, reordered_weights)

import torch

def tensor_parallel_allreduce_cost(
    message_element_count: int,
    bytes_per_element: int,
    tensor_parallel_world_size: int,
    link_bandwidth: float,
    per_hop_latency: float,
    collective_count: int,
) -> torch.Tensor:
    """
    Returns a float64 (2,) tensor: per-rank bytes and seconds across all collectives.
    """
    n = tensor_parallel_world_size
    message_bytes = message_element_count * bytes_per_element

    ring_steps = 2 * (n - 1)
    per_rank_bytes_per_collective = ring_steps * message_bytes / n if n > 0 else 0.0

    bandwidth_term = per_rank_bytes_per_collective / link_bandwidth
    latency_term = ring_steps * per_hop_latency
    time_per_collective = latency_term + bandwidth_term

    total_bytes = collective_count * per_rank_bytes_per_collective
    total_time = collective_count * time_per_collective

    return torch.tensor([total_bytes, total_time], dtype=torch.float64)

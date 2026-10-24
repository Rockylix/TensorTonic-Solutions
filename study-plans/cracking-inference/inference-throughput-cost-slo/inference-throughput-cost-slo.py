import torch

def select_inference_configuration(
    candidates: torch.Tensor,
    required_rate: float,
    max_p95_latency: float,
) -> tuple:
    """
    Returns (selected_index, hourly_cost): scalar int64 and float64 tensors; (-1, -1.0) if infeasible.
    """
    n = candidates.shape[0]
    best_idx = -1
    best_key = None

    for i in range(n):
        gpu, batch, throughput, p95, cost = candidates[i].tolist()
        if throughput >= required_rate and p95 <= max_p95_latency:
            key = (cost, gpu, batch, i)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = i

    if best_idx == -1:
        return torch.tensor(-1, dtype=torch.int64), torch.tensor(-1.0, dtype=torch.float64)

    return (
        torch.tensor(best_idx, dtype=torch.int64),
        torch.tensor(candidates[best_idx, 4].item(), dtype=torch.float64),
    )

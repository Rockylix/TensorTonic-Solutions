import torch

def latency_percentiles(latencies: torch.Tensor) -> torch.Tensor:
    """
    Returns a floating tensor of shape (3,), ordered P50, P95, P99.
    """
    q = torch.tensor([0.5, 0.95, 0.99], dtype=latencies.dtype)
    return torch.quantile(latencies, q)

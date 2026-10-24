import torch

def compute_serving_metrics(
    arrival_times: torch.Tensor,
    first_token_times: torch.Tensor,
    token_timestamps: torch.Tensor,
    output_token_counts: torch.Tensor,
) -> tuple:
    """
    Returns (ttft, tpot, mean_itl, throughput): three floating (N,) tensors and one floating scalar tensor.
    """
    num_requests = arrival_times.shape[0]

    ttft = first_token_times - arrival_times

    tpot = torch.zeros(num_requests, dtype=token_timestamps.dtype)
    mean_itl = torch.zeros(num_requests, dtype=token_timestamps.dtype)
    last_valid_times = torch.zeros(num_requests, dtype=token_timestamps.dtype)

    for r in range(num_requests):
        n = int(output_token_counts[r].item())
        valid = token_timestamps[r, :n]
        last_valid_times[r] = valid[-1]

        if n > 1:
            gaps = valid[1:] - valid[:-1]
            mean_itl[r] = gaps.mean()
            tpot[r] = (valid[-1] - valid[0]) / (n - 1)

    total_tokens = output_token_counts.sum().to(token_timestamps.dtype)
    window_start = arrival_times.min()
    window_end = last_valid_times.max()
    aggregate_throughput = total_tokens / (window_end - window_start)

    return ttft, tpot, mean_itl, aggregate_throughput

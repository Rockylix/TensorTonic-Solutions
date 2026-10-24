import torch

def simulate_disaggregated_serving(
    arrival_times: torch.Tensor,
    prompt_lengths: torch.Tensor,
    output_lengths: torch.Tensor,
    prefill_replica_rates: torch.Tensor,
    decode_replica_rates: torch.Tensor,
    kv_transfer_bandwidth: float,
    fixed_transfer_latency: float,
) -> tuple:
    """
    Returns (assigned_replicas, ttft, completion_time): integer (N, 2) assignments
    and two floating (N,) absolute timestamp tensors, in seconds.
    """
    n = arrival_times.shape[0]
    p = prefill_replica_rates.shape[0]
    d = decode_replica_rates.shape[0]

    prefill_avail = [0.0] * p
    assigned_prefill = [0] * n
    prefill_end = [0.0] * n

    order_by_arrival = sorted(range(n), key=lambda i: (arrival_times[i].item(), i))
    for i in order_by_arrival:
        best = min(range(p), key=lambda r: (prefill_avail[r], r))
        start = max(arrival_times[i].item(), prefill_avail[best])
        dur = prompt_lengths[i].item() / prefill_replica_rates[best].item()
        end = start + dur
        prefill_avail[best] = end
        assigned_prefill[i] = best
        prefill_end[i] = end

    transfer_time = [prompt_lengths[i].item() / kv_transfer_bandwidth + fixed_transfer_latency for i in range(n)]
    kv_ready = [prefill_end[i] + transfer_time[i] for i in range(n)]

    decode_avail = [0.0] * d
    assigned_decode = [0] * n
    ttft = [0.0] * n
    completion = [0.0] * n

    order_by_ready = sorted(range(n), key=lambda i: (kv_ready[i], i))
    for i in order_by_ready:
        best = min(range(d), key=lambda r: (decode_avail[r], r))
        start = max(kv_ready[i], decode_avail[best])
        first_tok = start + 1.0 / decode_replica_rates[best].item()
        dur = output_lengths[i].item() / decode_replica_rates[best].item()
        end = start + dur
        decode_avail[best] = end
        assigned_decode[i] = best
        ttft[i] = first_tok
        completion[i] = end

    assigned_replicas = torch.tensor(list(zip(assigned_prefill, assigned_decode)), dtype=torch.int64)
    return assigned_replicas, torch.tensor(ttft, dtype=torch.float64), torch.tensor(completion, dtype=torch.float64)

import torch

def schedule_chunked_prefill(
    arrival_steps: torch.Tensor,
    prompt_tokens: torch.Tensor,
    decode_tokens: torch.Tensor,
    max_tokens_per_iteration: int,
    prefill_chunk_size: int,
) -> torch.Tensor:
    """
    Returns an int64 (E, 4) tensor: iteration, request index, phase (0 prefill, 1 decode), token count.
    """
    n = arrival_steps.shape[0]
    prefill_remaining = prompt_tokens.clone()
    decode_remaining = decode_tokens.clone()
    admitted = torch.zeros(n, dtype=torch.bool)
    in_decode = torch.zeros(n, dtype=torch.bool)
    done = torch.zeros(n, dtype=torch.bool)

    events = []
    iteration = 0
    num_done = 0

    while num_done < n:
        for i in range(n):
            if not admitted[i].item() and arrival_steps[i].item() <= iteration:
                admitted[i] = True

        budget = max_tokens_per_iteration

        # Decode has strict priority: every already-prefilled request gets its
        # one token before any budget is spent on prefill this iteration.
        for i in range(n):
            if budget <= 0:
                break
            if admitted[i].item() and in_decode[i].item() and not done[i].item():
                events.append([iteration, i, 1, 1])
                decode_remaining[i] -= 1
                budget -= 1
                if decode_remaining[i].item() == 0:
                    done[i] = True
                    num_done += 1

        # Remaining budget fills prefill chunks in FCFS (arrival, index) order.
        prefill_candidates = [
            i for i in range(n)
            if admitted[i].item() and not in_decode[i].item() and not done[i].item() and prefill_remaining[i].item() > 0
        ]
        prefill_candidates.sort(key=lambda i: (arrival_steps[i].item(), i))
        for i in prefill_candidates:
            if budget <= 0:
                break
            chunk = min(budget, prefill_chunk_size, int(prefill_remaining[i].item()))
            if chunk <= 0:
                continue
            events.append([iteration, i, 0, chunk])
            prefill_remaining[i] -= chunk
            budget -= chunk
            if prefill_remaining[i].item() == 0:
                in_decode[i] = True
                if decode_remaining[i].item() == 0:
                    done[i] = True
                    num_done += 1

        iteration += 1

    if events:
        return torch.tensor(events, dtype=torch.int64)
    return torch.zeros((0, 4), dtype=torch.int64)

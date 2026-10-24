import torch

def simulate_continuous_batching(
    arrival_steps: torch.Tensor,
    required_tokens: torch.Tensor,
    max_active_sequences: int,
) -> tuple:
    """
    Returns (active_schedule, completion_step): a bool (S, N) tensor and an integer (N,) tensor.
    """
    n = arrival_steps.shape[0]
    remaining = required_tokens.clone()
    admitted = torch.zeros(n, dtype=torch.bool)
    active = torch.zeros(n, dtype=torch.bool)
    completion_step = torch.full((n,), -1, dtype=torch.int64)

    schedule_rows = []
    step = 0
    num_completed = 0

    while num_completed < n:
        free_slots = max_active_sequences - int(active.sum().item())
        if free_slots > 0:
            candidates = [
                i for i in range(n)
                if not admitted[i].item() and arrival_steps[i].item() <= step
            ]
            candidates.sort(key=lambda i: (arrival_steps[i].item(), i))
            for i in candidates[:free_slots]:
                admitted[i] = True
                active[i] = True

        schedule_rows.append(active.clone())

        for i in range(n):
            if active[i]:
                remaining[i] -= 1
                if remaining[i].item() == 0:
                    completion_step[i] = step
                    active[i] = False
                    num_completed += 1

        step += 1

    active_schedule = torch.stack(schedule_rows, dim=0)
    return active_schedule, completion_step

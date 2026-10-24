import torch

def schedule_dynamic_batches(
    arrival_times: torch.Tensor,
    max_batch_size: int,
    max_queue_delay: float,
) -> tuple:
    """
    Returns (batch_assignment, batch_start_time): integer IDs and floating dispatch times, each shape (N,).
    """
    
    n = arrival_times.shape[0]
    order = sorted(range(n), key=lambda i: (arrival_times[i].item(), i))

    batch_assignment = torch.zeros(n, dtype=torch.int64)
    batch_start_time = torch.zeros(n, dtype=torch.float32)

    batch_id = 0
    current_batch = []
    deadline = None

    def dispatch(dispatch_time):
        nonlocal batch_id, current_batch, deadline
        for idx in current_batch:
            batch_assignment[idx] = batch_id
            batch_start_time[idx] = dispatch_time
        batch_id += 1
        current_batch = []
        deadline = None

    for idx in order:
        t = arrival_times[idx].item()

        if current_batch and t >= deadline:
            dispatch(deadline)

        if not current_batch:
            deadline = t + max_queue_delay

        current_batch.append(idx)

        if len(current_batch) == max_batch_size:
            dispatch(t)

    if current_batch:
        dispatch(deadline)

    return batch_assignment, batch_start_time
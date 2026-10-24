import torch

def estimate_inference_gpu_count(
    parameter_count: int,
    bytes_per_parameter: float,
    kv_cache_bytes: int,
    activation_bytes: int,
    runtime_overhead_fraction: float,
    usable_bytes_per_gpu: int,
) -> torch.Tensor:
    """
    Returns an int64 (2,) tensor: required bytes, minimum GPU count.
    """
    weight_bytes = parameter_count * bytes_per_parameter
    base_bytes = weight_bytes + kv_cache_bytes + activation_bytes
    total_required_bytes = math.ceil(base_bytes * (1.0 + runtime_overhead_fraction))
    min_gpu_count = math.ceil(total_required_bytes / usable_bytes_per_gpu)
    return torch.tensor([total_required_bytes, min_gpu_count], dtype=torch.int64)

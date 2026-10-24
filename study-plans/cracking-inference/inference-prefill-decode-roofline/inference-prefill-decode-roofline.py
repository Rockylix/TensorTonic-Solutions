import torch

def analyze_prefill_decode_roofline(
    d_model: int,
    num_layers: int,
    batch_size: int,
    prompt_length: int,
    generated_length: int,
    element_size: int,
    peak_flops: float,
    memory_bandwidth: float,
) -> torch.Tensor:
    """
    Returns a float64 (2, 5) tensor: prefill/decode rows; FLOPs, bytes, FLOP/byte, FLOP/s, seconds.
    """
    params = 12 * num_layers * d_model ** 2

    prefill_flops = 2 * params * batch_size * prompt_length
    prefill_bytes = params * element_size

    decode_flops = 2 * params * batch_size * generated_length
    decode_bytes = params * element_size * generated_length

    def phase_metrics(flops: float, bytes_moved: float):
        arithmetic_intensity = flops / bytes_moved
        attainable = min(peak_flops, memory_bandwidth * arithmetic_intensity)
        latency = max(flops / peak_flops, bytes_moved / memory_bandwidth)
        return [flops, bytes_moved, arithmetic_intensity, attainable, latency]

    rows = [
        phase_metrics(prefill_flops, prefill_bytes),
        phase_metrics(decode_flops, decode_bytes),
    ]
    return torch.tensor(rows, dtype=torch.float64)

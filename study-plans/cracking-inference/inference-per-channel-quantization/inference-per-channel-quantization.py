import torch

def per_channel_int8_quantize(
    x: torch.Tensor,
    channel_axis: int,
) -> tuple:
    """
    Returns (quantized, scale, dequantized): int8 codes, broadcastable float scales, float reconstruction.
    """
    if channel_axis < 0:
        channel_axis += x.dim()
    reduce_dims = [d for d in range(x.dim()) if d != channel_axis]
    abs_max = x.abs().amax(dim=reduce_dims, keepdim=True)

    scale = torch.where(abs_max == 0, torch.ones_like(abs_max), abs_max / 127.0)

    quantized = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    dequantized = quantized.float() * scale

    return quantized, scale, dequantized

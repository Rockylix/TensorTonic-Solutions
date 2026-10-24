import torch

def symmetric_int8_quantize(
    x: torch.Tensor,
) -> tuple:
    """
    Returns (quantized, scale, dequantized): int8 codes, scalar float scale, float reconstruction.
    """
    work = x if x.is_floating_point() else x.float()
    amax = work.abs().max()
    scale = amax / 127.0 if amax > 0 else torch.ones((), dtype=work.dtype, device=x.device)  # 标量浮点张量
    q = torch.clamp(torch.round(work / scale), -127, 127).to(torch.int8)  # round 为四舍六入五成双
    dequant = q.to(work.dtype) * scale                        # 反量化回浮点
    return q, scale, dequant

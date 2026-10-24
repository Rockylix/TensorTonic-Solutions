import torch

def groupwise_int4_quantize(
    weight: torch.Tensor,
    group_size: int,
) -> tuple:
    """
    Returns (codes, scales, dequantized): int8 codes, float row-group scales, float reconstruction.
    """
    rows,cols = weight.shape
    num_full_groups = cols// group_size
    full_cols = num_full_groups * group_size
    # 剩余的组
    remainder = cols - full_cols
    # 额外算一组
    num_groups = num_full_groups + (1 if remainder > 0 else 0)

    codes = torch.zeros(rows, cols, dtype=torch.int8)
    scales = torch.zeros(rows, num_groups, dtype=torch.float32)
    dequantized = torch.zeros(rows, cols, dtype=torch.float32)

    if full_cols > 0:
        full_part = weight[:, :full_cols].reshape(rows, num_full_groups, group_size)
        abs_max = full_part.abs().amax(dim=2)
        scale = torch.where(abs_max == 0, torch.ones_like(abs_max), abs_max / 7.0)
        q = torch.clamp(torch.round(full_part / scale.unsqueeze(-1)), -7, 7).to(torch.int8)

        codes[:, :full_cols] = q.reshape(rows, full_cols)
        scales[:, :num_full_groups] = scale
        dequantized[:, :full_cols] = (q.float() * scale.unsqueeze(-1)).reshape(rows, full_cols)

    if remainder > 0:
            partial = weight[:, full_cols:]
            abs_max = partial.abs().amax(dim=1)
            scale = torch.where(abs_max == 0, torch.ones_like(abs_max), abs_max / 7.0)
            q = torch.clamp(torch.round(partial / scale.unsqueeze(-1)), -7, 7).to(torch.int8)
    
            codes[:, full_cols:] = q
            scales[:, num_full_groups] = scale
            dequantized[:, full_cols:] = q.float() * scale.unsqueeze(-1)
    
    return codes, scales, dequantized
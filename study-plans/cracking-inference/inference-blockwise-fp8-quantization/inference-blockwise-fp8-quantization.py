import torch

def blockwise_fp8_quantize(
    x: torch.Tensor,
    row_block_size: int,
    col_block_size: int,
    fp8_max: float,
) -> tuple:
    """
    Returns (scaled, block_scales, dequantized), floating tensors with one scale per 2D block.
    """
    rows, cols = x.shape
    num_row_blocks = -(-rows // row_block_size)
    num_col_blocks = -(-cols // col_block_size)

    scaled = torch.zeros_like(x)
    block_scales = torch.zeros(num_row_blocks, num_col_blocks)
    dequantized = torch.zeros_like(x)

    for i in range(num_row_blocks):
        r0 = i * row_block_size
        r1 = min(r0 + row_block_size, rows)
        for j in range(num_col_blocks):
            c0 = j * col_block_size
            c1 = min(c0 + col_block_size, cols)

            block = x[r0:r1, c0:c1]
            abs_max = block.abs().max()
            scale = torch.tensor(1.0) if abs_max.item() == 0.0 else abs_max / fp8_max

            block_scaled = torch.clamp(block / scale, -fp8_max, fp8_max)

            scaled[r0:r1, c0:c1] = block_scaled
            block_scales[i, j] = scale
            dequantized[r0:r1, c0:c1] = block_scaled * scale

    return scaled, block_scales, dequantized

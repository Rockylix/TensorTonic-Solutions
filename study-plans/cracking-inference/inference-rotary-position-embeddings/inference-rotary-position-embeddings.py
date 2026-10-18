import torch

def apply_rotary_position_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> tuple:
    """
    Returns (rotated_query, rotated_key), float32 tensors with the input shapes.
    """
    # 在dimension 维度上 两对特征构成一次旋转
    d = query.shape[-1]
    print(d)
    half = d // 2

    # 因为现在总共有 d/2对特征，他们的index 就用half来表示 ，不同的i代表不同的特征对
    i = torch.arange(half,dtype=torch.float32,device = query.device)
    # 每个特征对旋转的速度 由base来作为基础决定  
    theta = 1/ (base**(2*i/d))

    pos = positions.float()
    angles = pos.unsqueeze(1) * theta.unsqueeze(0)
    print(angles)
    #[seq,half]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    print(cos.shape)

    def embedding(x:torch.Tensor) -> torch.Tensor:
        # [...,start:stop:stride] 保留前面所有维度，后面的从start开始 stop省略，步长为2
        x_even = x[...,0::2]
        print(x_even.shape)
        x_odd = x[...,1::2]
        # x_even [batch,seq,half]
        rotate_even = x_even*cos - x_odd*sin
        rotate_odd = x_even*sin + x_odd*cos
        out  = torch.empty_like(x)
        out[...,0::2] = rotate_even
        out[...,1::2] = rotate_odd
        return out

    return (embedding(query),embedding(key))
    

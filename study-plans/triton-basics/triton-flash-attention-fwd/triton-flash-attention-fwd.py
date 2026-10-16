import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, out_ptr,
    N_Q, N_KV, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # Write code here
    pid =tl.program_id(0)

    off_m = pid * BLOCK_M + tl.arange(0,BLOCK_M)
    off_d = tl.arange(0,BLOCK_DMODEL)
    off_n = tl.arange(0,BLOCK_N)

    ptr_q = Q_ptr + off_m[:,None]*D + off_d[None,:]
    mask_q = (off_m[:,None]< N_Q) & (off_d[None,:]<D)
    q = tl.load(ptr_q,mask=mask_q,other =0.0);

    # 维护flash attention 的 三个数 m ， l 和 o 输出
    m_i = tl.full((BLOCK_M,), -1.0e30, dtype = tl.float32)
    l_i = tl.zeros((BLOCK_M,),dtype = tl.float32)
    o_i = tl.zeros((BLOCK_M,BLOCK_DMODEL),dtype = tl.float32)
    
    for n in range(0 ,tl.cdiv(N_KV,BLOCK_N)):
        cur_n = n* BLOCK_N + off_n
        kv_mask = (cur_n[:,None]<N_KV) & (off_d[None,:]<D)
        ptr_k = K_ptr + cur_n[:,None]*D + off_d[None,:]
        ptr_v = V_ptr + cur_n[:,None]*D + off_d[None,:]

        k = tl.load(ptr_k,mask=kv_mask,other=0.0)
        v = tl.load(ptr_v,mask=kv_mask,other=0.0)

        s = tl.dot(q,tl.trans(k)) * scale
        #下面这一步 把 mask 掉的 0 要设置为 负无穷， 不然下面的 p= exp(s -m_new) 就会把mask掉的也算进来
        # (cur_n[None,:]<N_KV 是 因为乘的是K 的 trans  s 为（BLOCK_M,BLOCK_N), 而 cur_n 为（BLOCK_N,) 因此需要扩展行
        s = tl.where((cur_n[None,:]<N_KV),s,-1.0e30)

        m_new = tl.maximum(m_i,tl.max(s,axis = 1))
        alpha = tl.exp(m_i - m_new)
        # 更新分母        
        # 维护p 就是指数权重
        p = tl.exp(s - m_new[:,None])
        l_i = l_i*alpha + tl.sum(p,axis = 1)

        #更新分子 acc[BLOCK_M, BLOCK_D]  alpha (BLOCK_M,)
        # 原来的分子 纠正
        o_i = o_i * alpha[:,None]
        # 加上当前块的分子
        o_i += tl.dot(p,v)

        #更新最大值 m (BLOCK_M,)
        m_i = m_new

    out = o_i / l_i[:,None]
    ptr_o = out_ptr + off_m[:,None] * D  + off_d[None,:]
    tl.store(ptr_o,out,mask = mask_q)    
        
    
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the FlashAttention forward kernel: out = softmax(Q K^T / sqrt(D)) V."""
    N_Q, D = Q.shape
    N_KV = K.shape[0]
    scale = 1.0 / (D ** 0.5)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_DMODEL = 16
    while BLOCK_DMODEL < D:
        BLOCK_DMODEL *= 2
    grid = ((N_Q + BLOCK_M - 1) // BLOCK_M,)
    flash_attention_fwd_kernel[grid](
        Q, K, V, out,
        N_Q,N_KV,D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )
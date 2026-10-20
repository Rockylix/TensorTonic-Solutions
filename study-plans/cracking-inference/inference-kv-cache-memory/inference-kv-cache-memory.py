import torch

def kv_cache_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_query_heads: int,
    gqa_kv_heads: int,
    head_dim: int,
    mla_latent_dim: int,
    mla_rotary_key_dim: int,
    bytes_per_element: int,
) -> torch.Tensor:
    """
    Returns an int64 tensor of byte counts ordered [MHA, MQA, GQA, MLA].
    """
    # 每一层是 key + value 也就是2 乘 batch 乘seq 长度， 乘头数 乘 每个头的维度  乘上每个元素的数据大小  最后 乘层数
    MHA_kvcache = 2 * batch_size * seq_len * num_layers * num_query_heads * head_dim * bytes_per_element

    # MQA  所有query 对应一个kv 头
    MQA_kvcahe = 2 * batch_size * seq_len * num_layers * head_dim * bytes_per_element

    # GQA 
    GQA_kvcache = 2 * batch_size* seq_len * num_layers * gqa_kv_heads * head_dim * bytes_per_element

    #MLA  只用存低维矩阵， 加上
    MLA_kvcache = batch_size * seq_len * num_layers * (mla_latent_dim + mla_rotary_key_dim) * bytes_per_element

    print(MHA_kvcache,MQA_kvcahe,GQA_kvcache,MLA_kvcache)
    
    return torch.tensor((MHA_kvcache,MQA_kvcahe,GQA_kvcache,MLA_kvcache),dtype = int)

    
"""
Simple implementation of flash attention forward pass
"""

import torch
import triton 
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, O, 
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    N_CTX, # number of tokens
    D: tl.constexpr, # dimension of the head
    H: tl.constexpr, # number of heads
    BLOCK_N: tl.constexpr, # kv rows per tile
    BLOCK_M: tl.constexpr # query rows per tile
):
    # grid indices from triton process
    # query start
    start_m = tl.program_id(0)
    # combined (batch, head) pair start
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # initialize offsets
    qo_offs = off_z * stride_qz + off_h * stride_qh
    k_offs = off_z * stride_kz + off_h * stride_kh
    v_offs = off_z * stride_vz + off_h * stride_vh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) # updated inside loop
    offs_d = tl.arange(0, D)

    # load query tile
    q_ptrs = Q + qo_offs + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # online softmax values
    # max logit so far
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    # sum of exponentials (denominator)
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    # running numerator
    numer = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        cur_n = start_n + offs_n

        # load k and v tiles
        k_ptrs = K + k_offs + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + v_offs + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=cur_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=cur_n[:, None] < N_CTX, other=0.0)

        # dot prod for attention
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # causal + bounds mask
        mask = (offs_m[:, None] >= cur_n[None, :]) & (cur_n[None, :] < N_CTX)
        qk = tl.where(mask, qk, float("-inf"))

        # online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.max(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p_curr = tl.exp(qk - m_new)
        denom = denom * alpha + tl.sum(p_curr, axis=1)
        numer = numer * alpha[:, None] + tl.dot(p_curr, v)
        m_i = m_new
    
    numer = numer/denom[:, None]
    out_ptrs = O + qo_offs + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, numer, mask=offs_m[:, None] < N_CTX)

def _fwd_attention(q, k, v, sm_scale):
    # B = num batches
    # H = num heads
    # S = num tokens
    # D = head dimension
    B, H, S, D = q.shape

    out = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _fwd_kernel[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N_CTX = S,
        D = D,
        H = H,
        BLOCK_N = BLOCK_N,
        BLOCK_M = BLOCK_M
    )

    return out


def naive_attention(q, k, v, sm_scale):
    """Reference causal attention using PyTorch ops."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    causal_mask = torch.tril(
        torch.ones((q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.bool)
    )
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)

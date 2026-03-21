"""
Simple implementation of cartesian pope kernel forward pass
"""

import argparse

import torch
import torch.nn.functional as F
import triton 
import triton.language as tl

@triton.jit
def pope_fwd_kernel(
    Q, K, V, O, 
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    N_CTX, # number of tokens
    V_DIM: tl.constexpr, # dimension of V
    QK_DIM: tl.constexpr, # dimension of Q and K
    H: tl.constexpr, # number of heads
    BLOCK_N: tl.constexpr, # kv rows per tile
    BLOCK_M: tl.constexpr # query rows per tile
):
    """
    kernel for forward pass of flash attention using PoPE positional embedding
    """

    # grid indices from triton process
    # query start
    start_m = tl.program_id(0)
    # combined (batch, head) pair start
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # initialize offsets
    q_offs = off_z * stride_qz + off_h * stride_qh
    k_offs = off_z * stride_kz + off_h * stride_kh
    v_offs = off_z * stride_vz + off_h * stride_vh
    o_offs = off_z * stride_oz + off_h * stride_oh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) # updated inside loop
    offs_d_qk = tl.arange(0, QK_DIM)
    offs_d_v = tl.arange(0, V_DIM)

    # load query tile
    q_ptrs = Q + q_offs + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # online softmax values
    # max logit so far
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    # sum of exponentials (denominator)
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    # running numerator
    numer = tl.zeros([BLOCK_M, V_DIM], dtype=tl.float32)
    valid_q = offs_m < N_CTX

    for start_n in range(0, N_CTX, BLOCK_N):
        cur_n = start_n + offs_n

        # load k and v tiles
        k_ptrs = K + k_offs + cur_n[:, None] * stride_kn + offs_d_qk[None, :] * stride_kk
        v_ptrs = V + v_offs + cur_n[:, None] * stride_vn + offs_d_v[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=cur_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=cur_n[:, None] < N_CTX, other=0.0)
        v = v.to(tl.float32)

        # dot prod for attention
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # causal + bounds mask
        mask = valid_q[:, None] & (offs_m[:, None] >= cur_n[None, :]) & (cur_n[None, :] < N_CTX)
        qk = tl.where(mask, qk, float("-inf"))

        # online softmax
        m_ij = tl.where(valid_q, tl.max(qk, axis=1), m_i)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(valid_q, tl.exp(m_i - m_new), 0.0)
        p_curr = tl.where(mask, tl.exp(qk - m_new[:, None]), 0.0)
        denom = denom * alpha + tl.sum(p_curr, axis=1)
        numer = numer * alpha[:, None] + tl.dot(p_curr, v)
        m_i = m_new
    
    numer = tl.where(valid_q[:, None], numer / denom[:, None], 0.0)
    out_ptrs = O + o_offs + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_ok
    tl.store(out_ptrs, numer, mask=valid_q[:, None])

def compute_theta(d, base=10000.0, device='cuda'):
    """
    theta_c = base ^ (c/d)
    """
    return base ** (torch.arange(d, device=device, dtype=torch.float32) / d)

def cartesian_pope(q, k, theta, delta=None):
    """
    compute 2d expansions for q and k
    """
    S = q.shape[-2]
    
    pos = torch.arange(S, device=q.device, dtype=torch.float32)
    phi_q = pos[:, None] * theta[None, :]
    phi_k = phi_q if delta is None else phi_q + delta[None, :]

    mu_q = F.softplus(q)
    mu_k = F.softplus(k)

    q_real = mu_q * torch.cos(phi_q)
    q_imag = mu_q * torch.sin(phi_q)
    k_real = mu_k * torch.cos(phi_k)
    k_imag = mu_k * torch.sin(phi_k)

    q_2d = torch.cat([q_real, q_imag], dim=-1)
    k_2d = torch.cat([k_real, k_imag], dim=-1)

    return q_2d.to(q.dtype), k_2d.to(k.dtype)

def pope_fwd_attention(q, k, v, sm_scale):
    """
    computes PoPE forward pass using pope_fwd_kernel
    """
    # B = num batches
    # H = num heads
    # S = num tokens
    # d = v dimension
    B, H, S, d_v = v.shape
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]

    # compute theta
    theta = compute_theta(d_v, device=q.device)

    # compute 2d expansion of q and k
    q_2d, k_2d = cartesian_pope(q, k, theta)

    # get last dimension of q and k, should be 2x d
    d_qk = q_2d.shape[-1]
    assert d_qk == 2 * d_v

    out = torch.empty_like(v)

    # The expanded q/k tiles drive shared-memory usage here, not v's head dim.
    # On T4-class GPUs, QK_DIM=128 with 64x64 tiles exceeds the 64 KiB limit.
    BLOCK_M = 32 if d_qk >= 128 else 64
    BLOCK_N = 32 if d_qk >= 128 else 64

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    pope_fwd_kernel[grid](
        q_2d, k_2d, v, out,
        sm_scale,
        q_2d.stride(0), q_2d.stride(1), q_2d.stride(2), q_2d.stride(3),
        k_2d.stride(0), k_2d.stride(1), k_2d.stride(2), k_2d.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N_CTX = S,
        V_DIM = d_v,
        QK_DIM = d_qk,
        H = H,
        BLOCK_N = BLOCK_N,
        BLOCK_M = BLOCK_M
    )

    return out

def naive_pope_attention(q, k, v, theta_c, sm_scale, delta=None):
    """
    naive pope attention for correctness comparison
    materializes full S x S score matrix
    """
    q_cart, k_cart = cartesian_pope(q, k, theta_c, delta)
    scores = torch.matmul(q_cart, k_cart.transpose(-2, -1)) * sm_scale
    causal_mask = torch.tril(
        torch.ones((q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.bool)
    )
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(attn, v)


def main():
    parser = argparse.ArgumentParser(description="Benchmark simple PoPE attention kernel.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check-seq-len", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark Triton kernels.")

    device = "cuda"
    dtype = torch.float16
    torch.manual_seed(0)
    sm_scale = args.dim ** -0.5

    def make_inputs(seq_len):
        shape = (args.batch, args.heads, seq_len, args.dim)
        q = torch.randn(shape, device=device, dtype=dtype)
        k = torch.randn(shape, device=device, dtype=dtype)
        v = torch.randn(shape, device=device, dtype=dtype)
        theta = compute_theta(args.dim, device=device)
        return q, k, v, theta

    check_seq_len = min(args.seq_len, args.check_seq_len)
    q, k, v, theta = make_inputs(check_seq_len)
    expected = naive_pope_attention(q, k, v, theta, sm_scale)
    actual = pope_fwd_attention(q, k, v, sm_scale)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    q, k, v, theta = make_inputs(args.seq_len)
    baseline_ms = triton.testing.do_bench(
        lambda: naive_pope_attention(q, k, v, theta, sm_scale),
        warmup=args.warmup,
        rep=args.rep,
    )
    kernel_ms = triton.testing.do_bench(
        lambda: pope_fwd_attention(q, k, v, sm_scale),
        warmup=args.warmup,
        rep=args.rep,
    )

    print(
        f"simple_pope batch={args.batch} heads={args.heads} "
        f"seq_len={args.seq_len} dim={args.dim}"
    )
    print(
        f"naive={baseline_ms:.3f} ms kernel={kernel_ms:.3f} ms "
        f"speedup={baseline_ms / kernel_ms:.2f}x"
    )


if __name__ == "__main__":
    main()

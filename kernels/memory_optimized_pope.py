"""
Simple implementation of cartesian pope kernel forward pass
"""

import argparse

import torch
import torch.nn.functional as F
import triton 
import triton.language as tl

@triton.jit
def pope_fwd_kernel_optimized(
    Q, K, V, O,
    Cos, Sin,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_angle_s, stride_angle_d,
    N_CTX, # number of tokens
    D: tl.constexpr, # dimension of the head
    H: tl.constexpr, # number of heads
    BLOCK_N: tl.constexpr, # kv rows per tile
    BLOCK_M: tl.constexpr # query rows per tile
):
    """
    kernel for forward pass of flash attention using PoPE positional embedding
    performs rotations inside the kernel to halve memory usage
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
    offs_d = tl.arange(0, D)

    # load query tile
    q_ptrs = Q + q_offs + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # load trig tables for query positions
    q_cos_ptrs = Cos + offs_m[:, None] * stride_angle_s + offs_d[None, :] * stride_angle_d
    q_sin_ptrs = Sin + offs_m[:, None] * stride_angle_s + offs_d[None, :] * stride_angle_d
    q_cos = tl.load(q_cos_ptrs, mask=offs_m[:, None] < N_CTX, other = 0.0)
    q_sin = tl.load(q_sin_ptrs, mask=offs_m[:, None] < N_CTX, other = 0.0)

    # safe softplus implementation
    q_float = q.to(tl.float32)
    mu_q = tl.where(q_float > 20.0, q_float, tl.log(1.0 + tl.exp(q_float)))

    # compute polar coordinates in register, avoids wasted memory usage
    q_x = (mu_q * q_cos).to(q.dtype)
    q_y = (mu_q * q_sin).to(q.dtype)

    # online softmax values
    # max logit so far
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    # sum of exponentials (denominator)
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    # running numerator
    numer = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    valid_q = offs_m < N_CTX

    for start_n in range(0, N_CTX, BLOCK_N):
        cur_n = start_n + offs_n

        # load k
        k_ptrs = K + k_offs + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=cur_n[:, None] < N_CTX, other=0.0)


        # load trig tables for k
        k_cos_ptrs = Cos + cur_n[:, None] * stride_angle_s + offs_d[None, :] * stride_angle_d
        k_sin_ptrs = Sin + cur_n[:, None] * stride_angle_s + offs_d[None, :] * stride_angle_d
        k_cos = tl.load(k_cos_ptrs, mask=cur_n[:, None] < N_CTX, other = 0.0)
        k_sin = tl.load(k_sin_ptrs, mask=cur_n[:, None] < N_CTX, other = 0.0) 

        # softplus for k
        k_float = k.to(tl.float32)
        mu_k = tl.where(k_float > 20.0, k_float, tl.log(1.0 + tl.exp(k_float)))

        # compute polar coords for k
        k_x = (mu_k * k_cos).to(k.dtype)
        k_y = (mu_k * k_sin).to(k.dtype)

        # dot prod for attention
        qk_x = tl.dot(q_x, tl.trans(k_x))
        qk_y = tl.dot(q_y, tl.trans(k_y))
        qk = (qk_x + qk_y) * sm_scale

        # causal + bounds mask
        mask = valid_q[:, None] & (offs_m[:, None] >= cur_n[None, :]) & (cur_n[None, :] < N_CTX)
        qk = tl.where(mask, qk, float("-inf"))

        # load v
        v_ptrs = V + v_offs + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=cur_n[:, None] < N_CTX, other=0.0)
        v = v.to(tl.float32)

        # online softmax
        m_ij = tl.where(valid_q, tl.max(qk, axis=1), m_i)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(valid_q, tl.exp(m_i - m_new), 0.0)
        p_curr = tl.where(mask, tl.exp(qk - m_new[:, None]), 0.0)
        denom = denom * alpha + tl.sum(p_curr, axis=1)
        numer = numer * alpha[:, None] + tl.dot(p_curr, v)
        m_i = m_new
    
    numer = tl.where(valid_q[:, None], numer / denom[:, None], 0.0)
    out_ptrs = O + o_offs + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, numer, mask=valid_q[:, None])

def compute_theta(d, base=10000.0, device='cuda'):
    """
    theta_c = base ^ (c/d)
    """
    return base ** (torch.arange(d, device=device, dtype=torch.float32) / d)

def compute_angle_matrix(S, theta, delta=None, device='cuda'):
    """
    compute S x d angle matrix
    """
    
    pos = torch.arange(S, device=device, dtype=torch.float32)
    phi_q = pos[:, None] * theta[None, :]
    phi_k = phi_q if delta is None else phi_q + delta[None, :]

    q_cos = torch.cos(phi_q)
    q_sin = torch.sin(phi_q)
    k_cos = torch.cos(phi_k)
    k_sin = torch.sin(phi_k)

    return q_cos, q_sin, k_cos, k_sin

def compute_trig_tables(S, theta, device='cuda', dtype=torch.float16):
    """
    compute shared S x d trig tables for the common delta=None path
    """
    pos = torch.arange(S, device=device, dtype=torch.float32)
    phi = pos[:, None] * theta[None, :]
    return torch.cos(phi).to(dtype), torch.sin(phi).to(dtype)

def pope_fwd_attention(q, k, v, sm_scale):
    """
    computes PoPE forward pass using pope_fwd_kernel
    """
    # B = num batches
    # H = num heads
    # S = num tokens
    # d = v dimension
    B, H, S, D = v.shape
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]

    # compute theta
    theta = compute_theta(D, device=q.device)

    # compute shared trig tables once; q and k use the same positions in this path
    cos, sin = compute_trig_tables(S, theta, device=q.device, dtype=q.dtype)

    out = torch.empty_like(v)

    # T4 memory constraint
    BLOCK_M = 32 if D >= 128 else 64
    BLOCK_N = 32 if D >= 128 else 64

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    pope_fwd_kernel_optimized[grid](
        q, k, v, out,
        cos, sin,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        cos.stride(0), cos.stride(1),
        N_CTX = S,
        D = D,
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
    q_cos, q_sin, k_cos, k_sin = compute_angle_matrix(
        q.shape[-2], theta_c, delta=delta, device=q.device
    )

    mu_q = F.softplus(q)
    mu_k = F.softplus(k)
    q_x = (mu_q * q_cos).to(q.dtype)
    q_y = (mu_q * q_sin).to(q.dtype)
    k_x = (mu_k * k_cos).to(k.dtype)
    k_y = (mu_k * k_sin).to(k.dtype)

    scores = (
        torch.matmul(q_x, k_x.transpose(-2, -1))
        + torch.matmul(q_y, k_y.transpose(-2, -1))
    ) * sm_scale
    causal_mask = torch.tril(
        torch.ones((q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.bool)
    )
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(attn, v)


def main():
    parser = argparse.ArgumentParser(description="Benchmark optimized PoPE attention kernel.")
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
        f"optimized_pope batch={args.batch} heads={args.heads} "
        f"seq_len={args.seq_len} dim={args.dim}"
    )
    print(
        f"naive={baseline_ms:.3f} ms kernel={kernel_ms:.3f} ms "
        f"speedup={baseline_ms / kernel_ms:.2f}x"
    )


if __name__ == "__main__":
    main()

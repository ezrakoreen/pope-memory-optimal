import argparse

import torch
import triton

try:
    from .simple_flash_attention import naive_attention, fwd_attention
except ImportError:
    from simple_flash_attention import naive_attention, fwd_attention

def rotate_half(x):
    """
    flips every pair of elements and negates the second element of the pair
    [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]
    """

    assert x.size(-1) % 2 == 0
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y = torch.stack((-x2, x1), dim=-1)
    return y.flatten(-2)

def rope_freqs(S, d, base=10000, device='cuda'):
    """
    precomputes cos and sin for rope
    returns cos, sin (shape [S, d])
    """

    assert d % 2 == 0

    # need to half dimensionality
    i = torch.arange(d // 2, dtype=torch.float32, device=device)
    freqs = base ** (-2 * i / d)

    # expand dimensionality back out
    freqs = freqs.repeat_interleave(2)

    # create position tensor
    pos = torch.arange(S, dtype=torch.float32, device=device)

    # expand into (S, d) grid
    theta = torch.outer(pos, freqs)

    return torch.cos(theta), torch.sin(theta)

def simple_rope_attention(q, k, v, cos, sin, sm_scale):
    """
    simple attention forward pass with rope
    """
    # rotate q and k
    q_rot = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k_rot = (k * cos + rotate_half(k) * sin).to(k.dtype)

    # use regular attention on rotated q and k
    return fwd_attention(q_rot, k_rot, v, sm_scale)

def naive_rope_attention(q, k, v, cos, sin, sm_scale):
    """
    naive attention forward pass with rope
    """
    # rotate q and k
    q_rot = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k_rot = (k * cos + rotate_half(k) * sin).to(k.dtype)

    # use regular attention on rotated q and k
    return naive_attention(q_rot, k_rot, v, sm_scale)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RoPE attention kernel.")
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
        cos, sin = rope_freqs(seq_len, args.dim, device=device)
        return q, k, v, cos, sin

    check_seq_len = min(args.seq_len, args.check_seq_len)
    q, k, v, cos, sin = make_inputs(check_seq_len)
    expected = naive_rope_attention(q, k, v, cos, sin, sm_scale)
    actual = simple_rope_attention(q, k, v, cos, sin, sm_scale)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    q, k, v, cos, sin = make_inputs(args.seq_len)
    baseline_ms = triton.testing.do_bench(
        lambda: naive_rope_attention(q, k, v, cos, sin, sm_scale),
        warmup=args.warmup,
        rep=args.rep,
    )
    kernel_ms = triton.testing.do_bench(
        lambda: simple_rope_attention(q, k, v, cos, sin, sm_scale),
        warmup=args.warmup,
        rep=args.rep,
    )

    print(
        f"rope batch={args.batch} heads={args.heads} "
        f"seq_len={args.seq_len} dim={args.dim}"
    )
    print(
        f"naive={baseline_ms:.3f} ms kernel={kernel_ms:.3f} ms "
        f"speedup={baseline_ms / kernel_ms:.2f}x"
    )


if __name__ == "__main__":
    main()

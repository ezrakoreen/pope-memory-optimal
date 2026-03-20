import torch
from .simple_flash_attention import naive_attention, _fwd_attention

def rotate_half(x):
    """
    flips every pair of elements and negates the second
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
    return _fwd_attention(q_rot, k_rot, v, sm_scale)

def naive_rope_attention(q, k, v, cos, sin, sm_scale):
    """
    naive attention forward pass with rope
    """
    # rotate q and k
    q_rot = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k_rot = (k * cos + rotate_half(k) * sin).to(k.dtype)

    # use regular attention on rotated q and k
    return naive_attention(q_rot, k_rot, v, sm_scale)

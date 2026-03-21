import pytest
import torch

from kernels.memory_optimized_pope import (
    compute_theta as optimized_compute_theta,
    naive_pope_attention as optimized_naive_pope_attention,
    pope_fwd_attention as optimized_pope_fwd_attention,
)
from kernels.rope import naive_rope_attention, rope_freqs, simple_rope_attention
from kernels.simple_flash_attention import fwd_attention, naive_attention
from kernels.simple_pope import (
    compute_theta as simple_compute_theta,
    naive_pope_attention as simple_naive_pope_attention,
    pope_fwd_attention as simple_pope_fwd_attention,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to run Triton kernel correctness tests.",
)


TEST_SHAPES = [
    (1, 2, 128, 64),
    (2, 4, 96, 64),
]


def make_inputs(shape, dtype=torch.float16, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    sm_scale = shape[-1] ** -0.5
    return q, k, v, sm_scale


@pytest.mark.parametrize("shape", TEST_SHAPES)
def test_simple_flash_attention_matches_naive(shape):
    q, k, v, sm_scale = make_inputs(shape)

    expected = naive_attention(q, k, v, sm_scale)
    actual = fwd_attention(q, k, v, sm_scale)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("shape", TEST_SHAPES)
def test_rope_attention_matches_naive(shape):
    q, k, v, sm_scale = make_inputs(shape)
    _, _, seq_len, dim = shape
    cos, sin = rope_freqs(seq_len, dim, device=q.device)

    expected = naive_rope_attention(q, k, v, cos, sin, sm_scale)
    actual = simple_rope_attention(q, k, v, cos, sin, sm_scale)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("shape", TEST_SHAPES)
def test_simple_pope_matches_naive(shape):
    q, k, v, sm_scale = make_inputs(shape)
    theta = simple_compute_theta(shape[-1], device=q.device)

    expected = simple_naive_pope_attention(q, k, v, theta, sm_scale)
    actual = simple_pope_fwd_attention(q, k, v, sm_scale)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("shape", TEST_SHAPES)
def test_optimized_pope_matches_naive(shape):
    q, k, v, sm_scale = make_inputs(shape)
    theta = optimized_compute_theta(shape[-1], device=q.device)

    expected = optimized_naive_pope_attention(q, k, v, theta, sm_scale)
    actual = optimized_pope_fwd_attention(q, k, v, sm_scale)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

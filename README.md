# RoPE + PoPE Attention Kernels

This repo is a progression of Triton attention kernels:

1. `kernels/simple_flash_attention.py`: a basic causal attention forward pass with online softmax that is based on FlashAttention v2.
2. `kernels/rope.py`: adds RoPE by rotating `q` and `k` before calling the flash attention kernel.
3. `kernels/simple_pope.py`: implements a simple Cartesian PoPE variant by expanding `q` and `k` into real and imaginary components, then running attention on the expanded representation.
4. `kernels/memory_optimized_pope.py`: keeps the same PoPE math, but avoids materializing the full expanded tensors by computing the PoPE coordinates inside the Triton kernel to minimize memory use.

`benchmarks/kernel_comparison.ipynb` benchmarks all four implementations for runtime and peak CUDA memory.

`tests/test.py` checks each kernel against a naive PyTorch reference on CUDA.

## Benchmark Summary

The benchmark notebook reports median runtime after warmup and peak additional CUDA memory allocated during a single forward pass.

### Median runtime (ms)

| implementation | 1x8x512x64 | 1x8x1024x64 | 1x8x2048x64 | 1x8x4096x64 | 2x8x1024x64 | 4x8x1024x64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flash | 1.327 | 3.900 | 8.207 | 30.961 | 4.294 | 7.817 |
| rope | 1.811 | 4.431 | 8.717 | 31.984 | 4.749 | 9.057 |
| simple_pope | 2.802 | 6.761 | 12.334 | 46.210 | 6.541 | 12.631 |
| memory_optimized_pope | 7.399 | 25.525 | 98.542 | 390.511 | 49.434 | 98.293 |

### Peak CUDA memory (MB)

| implementation | 1x8x512x64 | 1x8x1024x64 | 1x8x2048x64 | 1x8x4096x64 | 2x8x1024x64 | 4x8x1024x64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flash | 0.50 | 1.00 | 2.00 | 4.00 | 2.00 | 4.00 |
| rope | 3.75 | 7.50 | 15.00 | 30.00 | 14.50 | 28.50 |
| simple_pope | 11.13 | 22.25 | 44.51 | 89.02 | 44.25 | 88.25 |
| memory_optimized_pope | 0.63 | 1.25 | 2.50 | 5.00 | 2.25 | 4.25 |

## Why `memory_optimized_pope` is much slower but uses much less memory

`simple_pope` is faster because it computes the PoPE expansion once up front, materializes `q_2d` and `k_2d`, and then the attention kernel can reuse those tensors directly. However, those expanded tensors have shape `[B, H, S, 2D]`, so they are much larger than the original inputs. This substantially inflates peak memory use. `rope` has the same pattern, but to a lesser extent. It materializes rotated `q` and `k` with shape `[B, H, S, D]`, which is why it uses more memory than flash but much less than `simple_pope`.

`memory_optimized_pope` removes those large allocations. Instead of storing expanded PoPE tensors, it keeps only the original `q`, `k`, `v` plus shared trig tables and computes the PoPE coordinates inside the Triton kernel. That is why its peak memory stays close to flash attention.

The slowdown comes from recomputation. In `memory_optimized_pope`, each tile has to load trig values, apply `softplus`, form the PoPE coordinates, and do two dot products (`x` and `y`) instead of consuming precomputed tensors. More importantly, the key-side PoPE transform is recomputed every time a key tile is revisited by a new query tile. That avoids storing a `[B, H, S, 2D]` tensor, but it turns the one-time preprocessing cost into repeated work inside the attention loop, so runtime grows much more sharply with sequence length.

## Reproducing

Install the dependencies from `requirements.txt`, run `pytest tests/test.py` for correctness, and open `benchmarks/kernel_comparison.ipynb` to rerun the runtime and memory sweep.

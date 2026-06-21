
# CUDA Attention Performance Study

A case study in GPU performance engineering using attention as the workload:
why is attention slow, how do we prove it, what fixes it, and how close can
a self-built kernel get to production implementations?

## Hypothesis

## Phase 1: Naive Kernel — Baseline

## Phase 2: Profiling — Bottleneck Identification

Nsight Compute (`ncu`) hardware performance counters are blocked on Kaggle's
shared GPU environment (ERR_NVGPUCTRPERM — a known cloud platform restriction).
As a fallback, bottleneck analysis was done via manual roofline calculation
using Tesla T4 published specs (320 GB/s peak bandwidth, ~8.1 TFLOPS peak FP32).

**Naive Q@K^T kernel (seq_len=512, head_dim=64):**
- Memory traffic: ~135.3 MB (redundant reads, no data reuse across threads)
- FLOPs: ~33.6 million
- Arithmetic intensity: 0.248 FLOPs/byte
- T4 machine balance point: 25.3 FLOPs/byte
- **Conclusion: kernel is ~100x below the balance point → deeply memory-bound**

- Expected time if purely memory-bound: 0.423 ms
- Measured time (CUDA events): 1.03 ms
- Running at ~41% of theoretical peak memory bandwidth

## Phase 3: Tiled Kernel (FlashAttention-style)

## Phase 4: Tensor Core / WMMA Kernel

## Phase 5: Roofline Analysis

## Phase 6: Production Comparison (PyTorch SDPA, FlashAttention-2)

## Bonus Phase: Prefill vs Decode Study

## Lessons Learned

<!-- pushed from Kaggle -->

<!-- pushed from Kaggle -->

<!-- pushed from Kaggle -->

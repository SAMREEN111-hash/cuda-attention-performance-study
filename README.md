
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

Implemented shared-memory tiling for Q@K^T (TILE_SIZE=16, head_dim=64).
Verified bit-exact correctness against PyTorch (max diff = 0.0).

**Debugging note:** initial version had a bug where the shared-memory
store loop only copied back 16 of 64 columns per row (missing the
chunked loop that the load step used), causing 74% data corruption on a
load/store validation test. Fixed by mirroring the load loop's chunking
pattern in the store step.

**Performance (averaged over 3 runs per size, T4 GPU):**

| SEQ_LEN | Naive (ms) | Tiled (ms) | Ratio |
|---|---|---|---|
| 256 | 0.354 | 0.376 | 1.06x slower |
| 512 | 0.791 | 0.772 | 0.98x (tied) |
| 1024 | 2.524 | 2.627 | 1.04x slower |
| 2048 | 10.823 | 10.129 | 0.94x |

**Finding:** Naive run-to-run variance was substantial (up to 28% swing
at SEQ_LEN=2048 across identical runs on shared cloud GPU), making
single-run comparisons unreliable.

**Conclusion:** Reducing redundant global-memory reads via shared-memory
tiling does not automatically translate to a proportional speedup. At
head_dim=64 with TILE_SIZE=16, each block reuses loaded data only 16x
before discarding it and re-synchronizing — the synchronization and
load overhead largely offsets the memory-traffic savings at this scale.
Averaged results show tiled shows a small advantage at the largest
tested size (2048), but the gap is within the run-to-run noise floor
observed on this shared cloud GPU, so this should not be read as a
confident win — only as a direction worth pursuing further with
stronger optimizations (larger tiles, register-level reuse, warp-level
primitives, Tensor Cores).

## Phase 4: Tensor Core / WMMA Kernel

**Tensor Core Bring-Up (WMMA minimal kernel):**

Implemented a minimal WMMA kernel using NVIDIA Tensor Cores on a T4 GPU
(Turing, sm_75) before attempting real Q@K^T data.

Kernel configuration:
- Matrix size: 16×16 × 16×16
- Input precision: FP16 (half)
- Accumulator precision: FP32
- Execution model: one warp (32 threads) driving one Tensor Core operation

Key WMMA APIs validated: `wmma::fragment`, `wmma::load_matrix_sync`,
`wmma::mma_sync`, `wmma::store_matrix_sync`

Validation test: A = all ones, B = all ones (16×16 each).
Expected: every output element = 16 (sum of 16 products of 1×1).
Observed: all 256 output elements matched exactly, 0 errors.

**Result: PASS** — confirms the full WMMA pipeline (fragment load →
Tensor Core MMA → FP32 accumulation → store) works correctly on this
hardware before mapping real attention data onto it.


## Phase 5: Roofline Analysis

## Phase 6: Production Comparison (PyTorch SDPA, FlashAttention-2)

## Bonus Phase: Prefill vs Decode Study

## Lessons Learned

<!-- pushed from Kaggle -->

<!-- pushed from Kaggle -->

<!-- pushed from Kaggle -->

<!-- pushed from Kaggle -->

<!-- pushed from Kaggle -->

<!-- pushed from Kaggle -->

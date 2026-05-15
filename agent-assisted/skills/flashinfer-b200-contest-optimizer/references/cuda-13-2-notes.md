# CUDA 13.2 Notes for B200 Contest Optimization

## Why This Matters

CUDA 13.2 changes the optimization surface for Blackwell kernels in two ways:
- the official guidance surface is now centered on CUDA 13.2 + PTX ISA 9.2
- the current Nsight Compute release exposes better Blackwell and cluster-kernel diagnostics

Treat toolkit, driver, and profiler versions as part of the experiment, not background noise.

## Key Points

1. CUDA Toolkit 13.2 release date: `2026-03-05`.
2. Minimum driver for CUDA 13.2 on Linux x86_64: `>= 595.45.04`.
3. CUDA 13.2 compiler guidance explicitly points to PTX ISA `9.2` for semantic clarifications.
4. CUDA Programming Guide 13.2 documents Blackwell-specific performance levers relevant to this skill:
   - work stealing with Cluster Launch Control
   - thread-block clusters and Distributed Shared Memory
   - L2 persistence / access-policy windows
   - Programmatic Dependent Launch
   - memory synchronization domains
5. Blackwell tuning guidance confirms:
   - B200 keeps up to `256 KB` combined shared/L1/texture capacity
   - per-SM shared-memory capacity remains configurable up to `228 KB`
   - B200 supports a nonportable cluster size of `16`
6. Nsight Compute 2025.3 adds or improves support that matters for this workflow:
   - better Blackwell support
   - `launch__persisting_l2_cache_size`
   - improved cluster-kernel profiling behavior
   - new instruction-mix and scoreboard-dependency views

## Usage Guidance

- Pin toolkit, driver, and Nsight Compute versions in every round summary.
- When introducing inline PTX or CUDA 13.2-only launch features, verify the emitted SASS rather than trusting source-level intent.
- Use cluster features only when locality or tail-reduction gains exceed the occupancy cost.
- Use L2 persistence only when the reused window is stable enough to survive across many CTAs.
- Prefer fusion over PDL when fusion is feasible; use PDL only to overlap unavoidable multi-kernel pipelines.
- If the optimization hypothesis depends on a feature Triton cannot express, move the hotspot to CUDA instead of approximating it in Triton.

## Canonical Links

- CUDA Toolkit 13.2 Release Notes:
  https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- CUDA Toolkit 13.2 PDF Release Notes:
  https://docs.nvidia.com/cuda/pdf/CUDA_Toolkit_Release_Notes.pdf
- CUDA C++ Programming Guide 13.2:
  https://docs.nvidia.com/cuda/cuda-programming-guide/
- Blackwell Tuning Guide 13.2:
  https://docs.nvidia.com/cuda/blackwell-tuning-guide/
- PTX ISA 9.2:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- Nsight Compute 2025.3 Release Notes:
  https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/updates-2025-3.html

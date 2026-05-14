# B200 / Blackwell Notes for Kernel Tuning

Use this as an architecture-focused checklist.

## Hardware Notes

- B200 belongs to Blackwell (compute capability 10.0 family).
- Blackwell tuning guide highlights:
  - Large L2 capacity in Blackwell family (GB200 up to 126MB).
  - Unified Shared/L1/Texture capacity on B200-class devices: up to 256KB combined.
  - Shared-memory carveout is runtime-tunable via `cudaFuncSetAttribute(..., cudaFuncAttributePreferredSharedMemoryCarveout, ...)`.
  - Thread-block clusters are supported; B200 allows a nonportable cluster size of `16` with `cudaFuncAttributeNonPortableClusterSizeAllowed`.
  - Distributed Shared Memory (DSMEM) lets thread blocks in the same cluster read, write, and atomically update each other's shared memory.

## Practical Implications

1. Decode kernels with tiny batch often under-occupy GPU.
- Increase grid parallelism via row/head tiling.
- Avoid single-warp long serial loops unless reuse is extremely high.

2. L2-aware design matters.
- Keep reused vectors and scalar parameters cache-friendly.
- Evaluate persistence settings when reuse spans many thread blocks.
- Prefer L2 persistence for read-mostly metadata or reuse windows that survive across many CTAs in the same regime.

3. Shared memory is not always a win.
- Use it when data reuse outweighs synchronization cost.
- Re-check with p95 latency after adding `__syncthreads()`.
- If reuse crosses CTA boundaries, compare global-memory exchange against clusters + DSMEM before spending more time on local-shmem-only designs.

4. Tune launch bounds with occupancy metrics.
- Sweep 1/2/4 min blocks per SM where reasonable.
- Use Nsight metrics to confirm stall reduction instead of assuming gains.

5. Tail effects matter on irregular kernels.
- When the benchmark shows late-launch SM underfill, consider persistent work queues or Cluster Launch Control instead of endlessly retuning tile sizes.

6. Higher carveout is not free.
- Above-default shared-memory carveouts and dynamic shared-memory allocations above 48KB can unlock reuse, but often reduce occupancy. Always re-check active clusters / occupancy after changing carveout.

## Reference

- Blackwell Tuning Guide: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html

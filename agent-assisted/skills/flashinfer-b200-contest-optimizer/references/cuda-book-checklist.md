# CUDA Book Checklist (Applied to B200 Contest Kernels)

Use this concise checklist before calling a candidate “best”.

## 1) Occupancy and Launch Configuration

- Check active warps/SM and eligible warps.
- Remember: higher occupancy is not always faster, but low occupancy usually hurts latency hiding.
- Tune block size and `__launch_bounds__` with measurement.

## 2) Memory Access

- Ensure coalesced global access.
- Prefer vectorized load/store when aligned.
- Reduce redundant global reads via register reuse and small shared-memory staging.

## 3) Synchronization and Divergence

- Remove unnecessary barriers.
- Minimize branch divergence in warp hot paths.

## 4) Arithmetic Path

- Evaluate fast intrinsics only where tolerated by correctness gates.
- Use fused math patterns when instruction count drops and error stays acceptable.

## 5) Measurement Discipline

- Compare avg + p95 latency, not only peak speedup.
- Repeat best run at least once.
- Keep per-run artifacts and promote only reproducible winners.

## Reference

- CUDA C++ Best Practices Guide:
  https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf

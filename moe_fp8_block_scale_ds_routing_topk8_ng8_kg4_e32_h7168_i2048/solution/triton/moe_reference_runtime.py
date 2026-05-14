"""Build helpers for the vendored MoE CUDA extension.

The reference implementation originally embedded CUDA/C++ sources as Python
strings. Keep the same kernels in standalone source files, but still route the
build through `load_inline` so PyTorch generates the expected Python bindings
for the exported wrapper functions.
"""

from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load_inline


@lru_cache(maxsize=1)
def get_moe_runtime_lib():
    root = Path(__file__).parent
    return load_inline(
        name="flashinfer_moe_runtime_ext",
        cpp_sources=(root / "moe_reference_runtime.cpp").read_text(),
        cuda_sources=(root / "moe_reference_runtime.cu").read_text(),
        functions=[
            "fusedRoutePermuteCopyIntoWrapper",
            "buildPaddedOffsetsWrapper",
            "padFp8HiddenScaleWrapper",
            "transposeScaleMnToKWrapper",
            "unpadBf16RowsWrapper",
            "scaleBf16RowsByWeightWrapper",
            "scatterAddWrapper",
            "actQuantWrapper",
            "reduceAddWrapper",
        ],
        extra_cflags=["-O3", "-march=native"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )

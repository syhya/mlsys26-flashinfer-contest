import torch
import triton
import triton.language as tl

# Check available FP8 dtypes in Triton
print(dir(tl))
print(hasattr(tl, 'float8_e4m3fn'))
print(hasattr(tl, 'float8_e4m3'))
print(hasattr(tl, 'float8_e4m3fnuz'))
print(hasattr(tl, 'float8_e5m2'))

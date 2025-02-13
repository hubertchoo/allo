import allo
from allo.ir.types import int32
import numpy as np

M, N, K = 32, 32, 32

def gemm(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
    C: int32[M, N] = 0
    for i, j in allo.grid(M, N):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C
    
s = allo.customize(gemm)
s.reorder("k", "j")
s.buffer_at(s.C, axis="i")
s.pipeline("j")
code = s.build(target="vhls")
mod = s.build(target="vitis_hls", mode="csyn_xsim", project="gemm_xsim.prj")

A = np.random.randint(0, 10, size=(M, K), dtype=np.int32)
B = np.random.randint(0, 10, size=(K, N), dtype=np.int32)
output = np.zeros((M, N)).astype(np.int32)
mod(A, B, output, syn=False)
print(output)

expected = np.dot(A, B)
abs_diff = np.abs(output - expected)
rel_diff = abs_diff / (np.abs(expected) + 1e-10)
max_abs_diff = np.max(abs_diff)
max_rel_diff = np.max(rel_diff)
print(f"Max Absolute Difference: {max_abs_diff}")
print(f"Max Relative Difference: {max_rel_diff}")
np.testing.assert_allclose(output, expected)
import allo
from allo.ir.types import float32
import numpy as np
from allo.backend.pyxsi_ip import ParallelIPModuleCollection, SequentialIPModuleCollection, IPCollectionModeContext
import os

file_name = os.path.splitext(os.path.basename(__file__))[0]
new_dir = f"{file_name}_build"
os.makedirs(new_dir, exist_ok=True)
os.chdir(new_dir)
print(f"Current working directory: {os.getcwd()}")

M, N, K = 32, 32, 32

def gemm(A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
    C: float32[M, N] = 0.0
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

A = np.random.uniform(0, 10, size=(M, K)).astype(np.float32)
B = np.random.uniform(0, 10, size=(M, K)).astype(np.float32)
C = np.random.uniform(0, 10, size=(M, K)).astype(np.float32)
D = np.random.uniform(0, 10, size=(M, K)).astype(np.float32)

output = np.zeros((M, N)).astype(np.float32)
output2 = np.zeros((M, N)).astype(np.float32)

with IPCollectionModeContext():
    ip_collection = ParallelIPModuleCollection(
        mod(A, B, output, syn=True),
        mod(C, D, output2, syn=False)
        )
    
ip_collection()

print("PARALLEL TEST")
print("========================================")
print(np.dot(A, B))
print(output)
np.testing.assert_allclose(output, np.dot(A, B), rtol=1e-6, atol=1e-3)
print("-----------------------------")
print(np.dot(C, D))
print(output2)
np.testing.assert_allclose(output2, np.dot(C, D), rtol=1e-6, atol=1e-3)

##############################################################

output = np.zeros((M, N)).astype(np.float32)
output2 = np.zeros((M, N)).astype(np.float32)

with IPCollectionModeContext():
    ip_collection = SequentialIPModuleCollection(
        mod(A, B, output, syn=False),
        mod(C, output, output2, syn=False)
        )
    
ip_collection()

print("SEQUENTIAL TEST")
print("========================================")
print(np.dot(A, B))
print(output)
np.testing.assert_allclose(output, np.dot(A, B), rtol=1e-6, atol=1e-3)
print("-----------------------------")
print(np.dot(C, np.dot(A, B)))
print(output2)
np.testing.assert_allclose(output2, np.dot(C, np.dot(A, B)), rtol=1e-6, atol=1e-3)

import allo
from allo.ir.types import int32
import numpy as np
from allo.backend.pyxsi_ip import ParallelIPModuleCollection, SequentialIPModuleCollection, IPCollectionModeContext

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
C = np.random.randint(0, 10, size=(M, K), dtype=np.int32)
D = np.random.randint(0, 10, size=(K, N), dtype=np.int32)

output = np.zeros((M, N)).astype(np.int32)
output2 = np.zeros((M, N)).astype(np.int32)

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
np.testing.assert_allclose(output, np.dot(A, B))
print("-----------------------------")
print(np.dot(C, D))
print(output2)
np.testing.assert_allclose(output2, np.dot(C, D))

##############################################################

output = np.zeros((M, N)).astype(np.int32)
output2 = np.zeros((M, N)).astype(np.int32)

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
np.testing.assert_allclose(output, np.dot(A, B))
print("-----------------------------")
print(np.dot(C, np.dot(A, B)))
print(output2)
np.testing.assert_allclose(output2, np.dot(C, np.dot(A, B)))

import allo
from allo.ir.types import float32, int32
import numpy as np
from allo.backend.pyxsi_ip import ParallelIPModuleCollection
# from allo.backend.pyxsi_ip import SequentialIPModuleCollection
import os

file_name = os.path.splitext(os.path.basename(__file__))[0]
new_dir = f"{file_name}_build"
os.makedirs(new_dir, exist_ok=True)
os.chdir(new_dir)
print(f"Current working directory: {os.getcwd()}")

################ Model_1 float*float->float ################
M, N, K = 32, 32, 32
def gemm_float(A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
    C: float32[M, N] = 0.0
    for i, j in allo.grid(M, N):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C
    
s_ft = allo.customize(gemm_float)
s_ft.reorder("k", "j")
s_ft.buffer_at(s_ft.C, axis="i")
s_ft.pipeline("j")
code_ft = s_ft.build(target="vhls")
mod_ft = s_ft.build(target="vitis_hls", mode="csyn_xsim", project="gemm_float_xsim.prj")

################ Model_2 int*int->int ################
M, N, K = 32, 32, 32
def gemm_int(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
    C: int32[M, N] = 0
    for i, j in allo.grid(M, N):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C

s_int = allo.customize(gemm_int)
s_int.reorder("k", "j")
s_int.buffer_at(s_int.C, axis="i")
s_int.pipeline("j")
code_int = s_int.build(target="vhls")
mod_int = s_int.build(target="vitis_hls", mode="csyn_xsim", project="gemm_int_xsim.prj")

################ Model_3 float*int->float ################
M, N, K = 32, 32, 32
def gemm_mix(A: float32[M, K], B: int32[K, N]) -> float32[M, N]:
    C: float32[M, N] = 0
    for i, j in allo.grid(M, N):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C

s_mix = allo.customize(gemm_mix)
s_mix.reorder("k", "j")
s_mix.buffer_at(s_mix.C, axis="i")
s_mix.pipeline("j")
code_mix = s_mix.build(target="vhls")
mod_mix = s_mix.build(target="vitis_hls", mode="csyn_xsim", project="gemm_mix_xsim.prj")

M, N, K = 32, 32, 32
A = np.random.uniform(0, 10, size=(M, K)).astype(np.float32)
B = np.random.uniform(0, 10, size=(M, K)).astype(np.float32)
output1 = np.zeros((M, N)).astype(np.float32)

M, N, K = 32, 32, 32
C = np.random.randint(0, 10, size=(M, K), dtype=np.int32)
D = np.random.randint(0, 10, size=(K, N), dtype=np.int32)
output2 = np.zeros((M, N)).astype(np.int32)

M, N, K = 32, 32, 32
E = np.random.uniform(0, 10, size=(M, K)).astype(np.float32)
F = np.random.randint(0, 10, size=(M, K), dtype=np.int32)
output3 = np.zeros((M, N)).astype(np.float32)

ip_collection = ParallelIPModuleCollection(
    mod_ft(A, B, output1, syn=False, num_mod=3),
    mod_int(C, D, output2, syn=False, num_mod=3),
    mod_mix(E, F, output3, syn=False, num_mod=3),
    )
    
ip_collection()

# print("PARALLEL TEST")
# print("\n========================================\n")
# print(np.dot(A, B))
# print(output1)
# np.testing.assert_allclose(output1, np.dot(A, B), rtol=1e-6, atol=1e-3)
# print("\n----------------------------------------\n")
# print(np.dot(C, D))
# print(output2)
# np.testing.assert_allclose(output2, np.dot(C, D), rtol=1e-6, atol=1e-3)
# print("\n----------------------------------------\n")
# print(np.dot(E, F))
# print(output3)
# np.testing.assert_allclose(output3, np.dot(E, F), rtol=1e-6, atol=1e-3)

# output4 = np.zeros((M, N)).astype(np.int32)
# output5 = np.zeros((M, N)).astype(np.float32)
# output6 = np.zeros((M, N)).astype(np.float32)

# ip_collection = SequentialIPModuleCollection(
#     mod_int(C, D, output4, syn=False, num_mod=3),
#     mod_mix(A, output4, output5, syn=False, num_mod=3),
#     mod_ft(B, output5, output6, syn=False, num_mod=3),
#     )
    
# ip_collection()

# print("SEQUENTIAL TEST")
# print("\n========================================\n")
# print(np.dot(C, D))
# print(output4)
# np.testing.assert_allclose(output4, np.dot(C, D))
# print("\n----------------------------------------\n")
# print(np.dot(A, np.dot(C, D)))
# print(output5)
# np.testing.assert_allclose(output5, np.dot(A, np.dot(C, D)))
# print("\n----------------------------------------\n")
# print(np.dot(B, np.dot(A, np.dot(C, D))))
# print(output6)
# np.testing.assert_allclose(output6, np.dot(B, np.dot(A, np.dot(C, D))))


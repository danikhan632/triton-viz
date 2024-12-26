import triton
import triton.language as tl
import triton_viz
from triton_viz.interpreter import record_builder
import torch


torch.manual_seed(42)



@triton_viz.trace
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Compute a block of the matrix product C = A @ B.

    Args:
        A_ptr, B_ptr, C_ptr: Pointers to the data of A, B, and C in memory.
        M, N, K: Global dimensions of the matrices:
                 A is (M x K), B is (K x N), C is (M x N).
        stride_am, stride_ak: Strides for matrix A.
        stride_bk, stride_bn: Strides for matrix B.
        stride_cm, stride_cn: Strides for matrix C.
        BLOCK_M, BLOCK_N, BLOCK_K: Tile/block sizes used by the kernel.
    """
    # Program IDs in the 2D grid.
    pid_m = tl.program_id(axis=0)  # block id along the M dimension
    pid_n = tl.program_id(axis=1)  # block id along the N dimension

    # Compute the starting index of the output tile in C
    # i.e., which row (m) and column (n) this kernel block should handle
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Create index ranges for the block's rows and columns
    #  - e.g. [m_start, m_start+1, ..., m_start+BLOCK_M-1]
    #  - same for columns
    rm = m_start + tl.arange(0, BLOCK_M)
    rn = n_start + tl.arange(0, BLOCK_N)

    # Create a 2D mesh to index the output tile of size (BLOCK_M, BLOCK_N)
    # We'll accumulate partial sums into `acc`
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in steps of BLOCK_K
    # Each iteration accumulates a partial result of shape (BLOCK_M, BLOCK_N)
    # from sub-blocks of A and B.
    for k_block_start in range(0, K, BLOCK_K):
        # Range of indices along K dimension for this block
        rk = k_block_start + tl.arange(0, BLOCK_K)

        # ------- Load tile from A -------
        # We are loading a (BLOCK_M x BLOCK_K) sub-tile from A.
        # The offset in rows is 'rm', the offset in columns is 'rk'.
        A_tile_ptrs = A_ptr + (rm[:, None] * stride_am) + (rk[None, :] * stride_ak)
        a_tile = tl.load(A_tile_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)

        # ------- Load tile from B -------
        # We are loading a (BLOCK_K x BLOCK_N) sub-tile from B.
        # The offset in rows is 'rk', the offset in columns is 'rn'.
        B_tile_ptrs = B_ptr + (rk[:, None] * stride_bk) + (rn[None, :] * stride_bn)
        b_tile = tl.load(B_tile_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        # ------- Compute partial matmul on the tile -------
        # a_tile: (BLOCK_M, BLOCK_K)
        # b_tile: (BLOCK_K, BLOCK_N)
        # partial_acc: (BLOCK_M, BLOCK_N)
        acc += tl.dot(a_tile, b_tile)

    # ------- Store back to C -------
    # The final tile (BLOCK_M x BLOCK_N) is stored in C at location (rm x rn).
    C_tile_ptrs = C_ptr + (rm[:, None] * stride_cm) + (rn[None, :] * stride_cn)
    # Write out the results
    # We must also ensure we do not write out-of-bounds
    tl.store(C_tile_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

def matmul(A, B, M, N, K, 
           BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, 
           device='cuda'):
    """
    Helper Python function to call the matmul_kernel.
    A, B are torch tensors on GPU (or triton.compile-time pointers).
    M, N, K are the sizes: A(M,K), B(K,N), output C(M,N).
    """

    import torch

    # Create an output tensor C on the same device
    C = torch.zeros((M, N), dtype=torch.float32, device=device)

    # Strides (in elements, not bytes)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Grid: how many blocks along M and N dimensions
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N
    )

    # Launch the kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

    return C


# Dimensions
M, N, K = 256, 256, 256

# Random input matrices on GPU
A = torch.randn((M, K), dtype=torch.float32, device='cuda')
B = torch.randn((K, N), dtype=torch.float32, device='cuda')

# Run Triton matmul
C = matmul(A, B, M, N, K)
triton_viz.launch()  # Launch visualization after matmul operation
# Check correctness against PyTorch
C_ref = A @ B
print("Max difference:", (C - C_ref).abs().max().item())

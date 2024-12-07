import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.interpreter import record_builder
BLOCK_SIZE_M = 8
BLOCK_SIZE_N = 8
BLOCK_SIZE_K = 8


def printc(obj, color="cyan"):
    color_code = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)


@triton_viz.trace
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, 
                  stride_bk, stride_bn, stride_cm, stride_cn):
    
    BLOCK_SIZE_M = tl.constexpr(16)
    BLOCK_SIZE_N = tl.constexpr(16)
    BLOCK_SIZE_K = tl.constexpr(16)

    # Calculate program ID and group size
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    num_pid_x = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_y = tl.cdiv(N, BLOCK_SIZE_N)


    # Calculate memory offsets for accessing matrix tiles
    offs_am = (pid_x * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_y * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator for the product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    
    for i in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        # printc(accumulator)

    # Store the results
    c = accumulator.to(tl.float16)
    offs_cm = pid_x * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_y * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Matrices must be contiguous"
    M, K = a.shape
    N = b.shape[1]

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Configure 2D grid for better parallelism
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def perform_matmul(device, M, N, K):
    sz=1
    # Generate random values between 200 and 1000
    a = 200 + 800 * torch.rand((32*sz, 48*sz), device=device, dtype=torch.float32)
    b = 200 + 800 * torch.rand((48*sz, 32*sz), device=device, dtype=torch.float32)
    printc(b,'green')

    c = matmul(a, b)

    triton_viz.launch()  # Launch visualization after matmul operation
    return a, b, c

if __name__ == "__main__":
    device = "cpu"  # You can change this to your specific device
    M, N, K = 16*1, 16*1, 16*1
    a, b, c = perform_matmul(device, M, N, K)
    
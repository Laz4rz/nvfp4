"""
educational implementations
"""

import torch


def quantize_row_blockwise(x: torch.Tensor, block_size: int=16, num_bits: int=4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x: 1D tensor, shape (K,), float32
    returns:
        q: 1D tensor, shape (K,), integer-valued
        s: 1D tensor, shape (K // block_size,), float32 scales
    """
    K = x.shape[0]
    assert K % block_size == 0, "x must be divisible by block_size"
    n_blocks = K // block_size

    qmin = - (2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    # .new_zeros creates a new tensor with same dtype and device
    q = x.new_zeros(K, dtype=torch.int8)
    s = x.new_zeros(n_blocks, dtype=torch.float32)

    for bid in range(0, n_blocks):
        block = x[bid*block_size:(bid+1)*block_size]
        amax = torch.max(torch.abs(block))
        # scale = amax / (qmax + 1e-6)
        scale = torch.where(amax > 0, amax/qmax, torch.ones_like(amax))
        q_block = torch.round(block/scale)
        q_block = torch.clip(q_block, qmin, qmax)
        q[bid*block_size:(bid+1)*block_size] = q_block
        s[bid] = scale
    
    return q, s

def dequantize_row_blockwise(q: torch.Tensor, s: torch.Tensor, return_type: torch.dtype=torch.float32) -> torch.Tensor:
    """
    q: 1D tensor, shape (K,), ints
    s: 1D tensor, shape (K // block_size,), scales
    return:
        y: 1D tensor, shape (K,), float32
    """
    K = q.shape[0]
    num_blocks = s.shape[0]
    assert K % num_blocks == 0, "row size is not divisible by block_size"
    block_size = K // num_blocks

    y = q.new_zeros(K, dtype=return_type)

    for bid in range(0, num_blocks):
        block = q[bid*block_size:(bid+1)*block_size]
        y[bid*block_size:(bid+1)*block_size] = block * s[bid]

    return y

def quantize_matrix_blockwise(A_full: torch.Tensor, block_size: int=16, num_bits: int=4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A_full: (M, K), float32
    returns:
        A_q: (M, K), integer-ish
        SFA: (M, K // block_size), scales per row and block
    """
    M, K = A_full.shape
    assert K % block_size == 0, "x must be divisible by block_size"
    n_blocks = K // block_size

    A_q = torch.zeros(M, K, dtype=torch.int8)
    SFA = torch.zeros(M, n_blocks, dtype=torch.float32)

    for rid in range(0, M):
        A_q[rid, :], SFA[rid, :] = quantize_row_blockwise(A_full[rid, :], block_size, num_bits)
    
    return A_q, SFA

def dequantize_matrix_blockwise(A_q: torch.Tensor, SFA: torch.Tensor, return_type: torch.dtype=torch.float32) -> torch.Tensor:
    M, K = A_q.shape
    num_blocks = SFA.shape[1]
    assert K % num_blocks == 0, "row size is not divisible by block_size"

    A_full = torch.zeros(M, K, dtype=return_type)

    for rid in range(0, M):
        A_full[rid, :] = dequantize_row_blockwise(A_q[rid, :], SFA[rid, :])

    return A_full

def to_nvfp4(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.Tensor([]), torch.Tensor([]) 

def scaled_gemm_naive(A_q, B_q, SFA, SFB) -> torch.Tensor:
    A_full = dequantize_matrix_blockwise(A_q, SFA)
    B_full = dequantize_matrix_blockwise(B_q, SFB)

    return A_full @ B_full.T

def scaled_gemm_fused(A_q, B_q, SFA, SFB) -> torch.Tensor:
    assert A_q.shape[1] == B_q.shape[1], "A and B shapes don't match for matmul"
    M, K = A_q.shape
    num_blocks = SFA.shape[1]
    assert K % num_blocks == 0, "row size is not divisible by block_size"
    assert num_blocks == SFB.shape[1], "SFA and SFB shapees don't match"
    N = B_q.shape[0]

    return torch.tensor([])

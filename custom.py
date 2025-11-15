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
    assert K // block_size, "x must be divisible by block_size"
    n_blocks = K // block_size

    qmin = - (2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    q = torch.zeros(K)
    s = torch.zeros(n_blocks)

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

def dequantize_row_blockwise(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
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

    y = torch.zeros(K)

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
    assert K // block_size, "x must be divisible by block_size"
    n_blocks = K // block_size

    A_q = torch.zeros(M, K)
    SFA = torch.zeros(M, n_blocks)

    for rid in range(0, M):
        A_q[rid, :], SFA[rid, :] = quantize_row_blockwise(A_full[rid, :], block_size, num_bits)
    
    return A_q, SFA

def dequantize_matrix_blockwise(A_q: torch.Tensor, SFA: torch.Tensor) -> torch.Tensor:
    M, K = A_q.shape
    num_blocks = SFA.shape[1]
    assert K % num_blocks == 0, "row size is not divisible by block_size"

    A_full = torch.zeros(M, K)

    for rid in range(0, M):
        A_full[rid, :] = dequantize_row_blockwise(A_q[rid, :], SFA[rid, :])

    return A_full

def to_nvfp4(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.Tensor([]), torch.Tensor([]) 

def scaled_gemm() -> torch.Tensor:
    return torch.Tensor([])

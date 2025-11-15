import unittest
import torch

from custom import (
    quantize_row_blockwise,
    dequantize_row_blockwise,
    quantize_matrix_blockwise,
    dequantize_matrix_blockwise,
)


class TestQuantizeDequantizeRowBlockwise(unittest.TestCase):
    def test_quantize_shapes_match(self):
        x = torch.randn(32, dtype=torch.float32)
        q, s = quantize_row_blockwise(x, block_size=16, num_bits=4)

        self.assertEqual(q.shape, x.shape)
        self.assertEqual(s.shape, torch.Size([32 // 16]))

    def test_roundtrip_approx_identity(self):
        x = torch.linspace(-1.0, 1.0, steps=32, dtype=torch.float32)
        q, s = quantize_row_blockwise(x, block_size=16, num_bits=4)
        x_hat = dequantize_row_blockwise(q, s)

        self.assertEqual(x_hat.shape, x.shape)
        # Allow some quantization error
        self.assertTrue(torch.allclose(x, x_hat, atol=0.15, rtol=0.0))

    def test_zero_block_stays_zero(self):
        x = torch.zeros(16, dtype=torch.float32)
        q, s = quantize_row_blockwise(x, block_size=16, num_bits=4)
        x_hat = dequantize_row_blockwise(q, s)

        self.assertTrue(torch.all(q == 0))
        self.assertTrue(torch.all(x_hat == 0))

    def test_roundtrip_better_with_bits(self):
        x = torch.rand(16, dtype=torch.float32)
        num_bits_l = [4, 8]
        mses = []
        for num_bits in num_bits_l:
            q, s = quantize_row_blockwise(x, block_size=16, num_bits=num_bits)
            x_hat = dequantize_row_blockwise(q, s)
            mses.append((x - x_hat).pow(2).mean().item())
            # Assert monotonic non-increase
            for a, b in zip(mses, mses[1:]):
                self.assertLessEqual(b, a + 1e-7)
        print("Row: MSEs for [4, 8] bits:", mses)


class TestMatrixQuantizeDequantizeBlockwise(unittest.TestCase):
    def test_quantize_shapes_match(self):
        M, K, block_size = 8, 64, 16
        A = torch.randn(M, K, dtype=torch.float32)
        A_q, SFA = quantize_matrix_blockwise(A, block_size=block_size, num_bits=4)
        self.assertEqual(A_q.shape, A.shape)
        self.assertEqual(SFA.shape, torch.Size([M, K // block_size]))

    def test_roundtrip_approx_identity(self):
        M, K, block_size = 4, 64, 16
        A = torch.linspace(-1.0, 1.0, steps=M*K, dtype=torch.float32).view(M, K)
        A_q, SFA = quantize_matrix_blockwise(A, block_size=block_size, num_bits=4)
        A_hat = dequantize_matrix_blockwise(A_q, SFA)
        self.assertEqual(A_hat.shape, A.shape)
        self.assertTrue(torch.allclose(A, A_hat, atol=0.15, rtol=0.0))

    def test_zero_rows_stay_zero(self):
        M, K, block_size = 3, 32, 16
        A = torch.zeros(M, K, dtype=torch.float32)
        A_q, SFA = quantize_matrix_blockwise(A, block_size=block_size, num_bits=4)
        A_hat = dequantize_matrix_blockwise(A_q, SFA)
        self.assertTrue(torch.all(A_q == 0))
        self.assertTrue(torch.all(A_hat == 0))
        self.assertTrue(torch.all(SFA == 1))

    def test_row_consistency(self):
        M, K, block_size, num_bits = 6, 64, 16, 4
        A = torch.randn(M, K, dtype=torch.float32)
        A_q, SFA = quantize_matrix_blockwise(A, block_size=block_size, num_bits=num_bits)
        for rid in range(M):
            q_row_ref, s_row_ref = quantize_row_blockwise(A[rid], block_size=block_size, num_bits=num_bits)
            self.assertTrue(torch.equal(q_row_ref, A_q[rid]))
            self.assertTrue(torch.equal(s_row_ref, SFA[rid]))

    def test_error_monotonic_with_bits(self):
        M, K, block_size = 4, 64, 16
        A = torch.randn(M, K, dtype=torch.float32)
        bit_settings = [4, 8]
        mses = []
        for bits in bit_settings:
            A_q, SFA = quantize_matrix_blockwise(A, block_size=block_size, num_bits=bits)
            A_hat = dequantize_matrix_blockwise(A_q, SFA)
            mses.append((A - A_hat).pow(2).mean().item())
        self.assertLessEqual(mses[1], mses[0] + 1e-7)
        print("Matrix: MSEs for [4, 8] bits:", mses)


if __name__ == "__main__":
    unittest.main()

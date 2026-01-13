import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication of lower triangular matrices
    using a custom CUDA kernel that exploits the triangular structure.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        
        # CUDA kernel for triangular matrix multiplication
        cuda_kernel_code = """
        #define TILE_SIZE 32
        
        __global__ void tril_matmul_kernel(
            const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            const int N)
        {
            __shared__ float As[TILE_SIZE][TILE_SIZE];
            __shared__ float Bs[TILE_SIZE][TILE_SIZE];
            
            int row = blockIdx.y * TILE_SIZE + threadIdx.y;
            int col = blockIdx.x * TILE_SIZE + threadIdx.x;
            
            // Only compute lower triangular elements
            if (row < col || row >= N || col >= N) {
                if (row < N && col < N) {
                    C[row * N + col] = 0.0f;
                }
                return;
            }
            
            float sum = 0.0f;
            int max_k = min(row, col) + 1;  // Triangular bound
            
            // Tile across k dimension, but only up to max_k
            for (int tile = 0; tile * TILE_SIZE < max_k; ++tile) {
                int k_start = tile * TILE_SIZE;
                
                // Load tile from A
                int k_a = k_start + threadIdx.x;
                if (row < N && k_a < N && k_a <= row) {
                    As[threadIdx.y][threadIdx.x] = A[row * N + k_a];
                } else {
                    As[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                // Load tile from B
                int k_b = k_start + threadIdx.y;
                if (col < N && k_b < N && k_b <= col) {
                    Bs[threadIdx.y][threadIdx.x] = B[k_b * N + col];
                } else {
                    Bs[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                __syncthreads();
                
                // Compute partial sum for this tile
                int k_end = min(k_start + TILE_SIZE, max_k);
                for (int k = k_start; k < k_end; ++k) {
                    sum += As[threadIdx.y][k - k_start] * Bs[k - k_start][threadIdx.x];
                }
                
                __syncthreads();
            }
            
            if (row < N && col < N) {
                C[row * N + col] = sum;
            }
        }
        """
        
        # C++ wrapper code
        cpp_wrapper_code = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        
        #define TILE_SIZE 32
        
        __global__ void tril_matmul_kernel(
            const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            const int N);
        
        torch::Tensor tril_matmul(torch::Tensor A, torch::Tensor B) {
            TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
            TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
            TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
            TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
            
            const int N = A.size(0);
            auto C = torch::zeros({N, N}, A.options());
            
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
            
            tril_matmul_kernel<<<grid, block>>>(
                A.data_ptr<float>(),
                B.data_ptr<float>(),
                C.data_ptr<float>(),
                N
            );
            
            return C;
        }
        """
        
        # Compile the CUDA kernel
        self.cuda_module = load_inline(
            name="tril_matmul_cuda",
            cpp_sources=[cpp_wrapper_code],
            cuda_sources=[cuda_kernel_code],
            functions=["tril_matmul"],
            with_cuda=True,
            verbose=False
        )
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        # Ensure tensors are contiguous and on CUDA
        A = A.contiguous()
        B = B.contiguous()
        
        # Call the custom CUDA kernel
        return self.cuda_module.tril_matmul(A, B)
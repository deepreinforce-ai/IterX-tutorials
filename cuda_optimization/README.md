# Task: CUDA Kernel Optimization
**Difficulty Level: ⭐⭐⭐ (3 stars)**

---

## Background

In this task, we will design optimized CUDA kernels to accelerate PyTorch operations. Deep learning workloads spend most of their time in matrix operations, convolutions, and other tensor computations. While PyTorch provides highly optimized implementations through cuBLAS and cuDNN, there are  significant opportunities for domain-specific optimizations that can significantly improve performance.

For illustration, we use **Level 1 Task 15 from KernelBench**: Lower Triangular Matrix Multiplication. Given two lower triangular matrices A and B, compute C = tril(A × B). This task exploits the triangular structure to skip unnecessary computation in the upper triangular region.

The `get_reward()` function returns a tuple of **`(reward, error_msg, details)`**:
- **`reward`**: The speedup ratio (`original_time / custom_time`) over the reference kernel. Returns `0` if correctness or security checks fail.
- **`error_msg`**: Empty string `""` on success, or a description of the failure.
- **`details`**: Reserved for extended execution metrics such as NVIDIA Nsight Compute (NCU) profiling data (currently returns `""`).

---

## The Evaluation Function

### Why Process Isolation is Needed

GPU optimization represents a category of tasks where generated code can be dangerous:

1. **Machine Corruption**: Bad CUDA code can crash the GPU driver, corrupt GPU memory, or even require a machine reboot
2. **Infinite Execution**: Bad kernels can hang indefinitely, blocking the evaluation pipeline

Therefore, we need **process isolation** to safely evaluate untrusted generated code.

### Two Evaluation Approaches

#### 1. Standard Evaluation (`eval_cuda.py`)

Direct in-process evaluation - simple but risky:

```python
from iterx_code.cuda_optimization.eval_cuda import get_reward

reward, error_msg = get_reward(
    custom_model_path="/path/to/code.py",
    work_dir="/path/to/work_dir",
    device_index=0
)
```

**Pros:** Simple, no network overhead, direct function call
**Cons:** If the code hangs or crashes, it takes down the entire evaluation process

#### 2. Server-Based Evaluation (`eval_cuda_server.py`)

Process-isolated evaluation using a Flask server:

```python
from iterx_code.cuda_optimization.eval_cuda_server import get_reward

reward, error_msg = get_reward(
    code_path="/path/to/code.py",
    work_dir="/path/to/work_dir",
    eval_url="http://localhost:5001",
    log_folder="/path/to/logs"
)
```

**How it works:**
1. **Start Server**: Launch `cuda_eval_flask_server.py` as a separate process
2. **Send Request**: POST the evaluation request to the server
3. **Monitor Output**: The server writes progress to a log file (JSON lines)
4. **Poll for Results**: Client monitors the log file for completion
5. **Timeout Handling**: If a stage takes too long, kill the server process
6. **Cleanup**: Kill the server before returning

**Pros:** 
- Safe: bad code only crashes the server process, not the main evaluation
- Timeout control: can kill the server if it hangs
- Monitoring: can track progress through stages

**Cons:** 
- Network overhead
- More complex setup

### Alternative Approaches (Not Implemented)

**Redis-based Queue**: Use Redis as a job queue where workers pick up evaluation tasks. This provides:
- Better scalability for multiple GPUs
- Built-in job timeout and retry mechanisms
- Centralized job management

However, the server-based approach was chosen for simplicity.



## Task Description

**Goal:** Design a CUDA kernel that accelerates the reference PyTorch operation. 

**Input:** 
- Reference model in `initial_code.py` defining `Model`, `get_inputs()`, `get_init_inputs()`
- Your optimized implementation defining `ModelNew`

**Output:**
- Numerically equivalent results to the reference
- Faster execution time

**Requirements:**
1. Define a `ModelNew` class with the same interface as the reference `Model`
2. Implement `forward()` method with identical input/output signature
3. Keep `get_inputs()` and `get_init_inputs()` functions exactly as in the reference
4. Maintain all hyperparameters (e.g., `M = 4096`) exactly as in the reference
5. Output must match reference within tolerance (atol=1e-2, rtol=1e-2)
6. No timing cheats (thread/stream/lazy injection)
7. Your code must be in Python.

**Things to Avoid:**
1. Creating new CUDA streams for computation
2. Spawning background threads
3. Returning lazy tensor subclasses
4. Infinite loops or blocking operations
5. Hardcoded test cases (must work for general inputs)

**Writing CUDA in Python:**
You can write CUDA kernels directly in Python files using `torch.utils.cpp_extension.load_inline`. This allows you to embed CUDA C++ code as a string and compile it at runtime:

**HOW TO USE load_inline FOR CUSTOM CUDA KERNELS**:
You can write CUDA code in python using load_inline.
1. **Define CUDA kernel and C++ wrapper** (Note: `load_inline` auto-generates `PYBIND11_MODULE` when you specify `functions`, so do NOT include it manually):
```python
from torch.utils.cpp_extension import load_inline

# CUDA kernel code
cuda_kernel_code = \"\"\"
__global__ void my_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
\"\"\"

# C++ wrapper with pybind11 binding
cpp_wrapper_code = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

// Declare the CUDA kernel
__global__ void my_kernel(const float* A, const float* B, float* C, int N);

// C++ wrapper function that launches the kernel
torch::Tensor my_kernel_launcher(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    my_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
\"\"\"

# Load and compile the CUDA module
self.cuda_module = load_inline(
    name="my_cuda_module",
    cpp_sources=[cpp_wrapper_code],
    cuda_sources=[cuda_kernel_code],
    functions=["my_kernel"],
    with_cuda=True,
    verbose=False
)
```

2. **Call the kernel from Python**:
```python
# Simply call like a regular Python function
C = self.cuda_module.my_kernel(A, B)
```

3. **For 2D matrix kernels (e.g., tiled matmul)**:
```python
cpp_wrapper_code = \"\"\"
#include <torch/extension.h>

__global__ void tril_matmul_kernel(const float* A, const float* B, float* C, int N);

torch::Tensor tril_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (N + 31) / 32);
    
    tril_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
```

---

## Reward Description

The reward is computed as:

```
reward = original_time / custom_time
```
---


### Initial Code (Reference)

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs C = A * B where A and B are lower triangular matrices. 
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        return torch.tril(torch.matmul(A, B))

M = 4096

def get_inputs():
    A = torch.randn(M, M)
    B = torch.randn(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []
```


---

## File Structure

```
iterx_code/cuda_optimization/
├── README.md                    # This file
├── eval_cuda.py                 # Standard evaluation (in-process)
├── eval_cuda_server.py          # Server-based evaluation client (process-isolated)
├── cuda_eval_flask_server.py    # Flask server for isolated evaluation
├── initial_code.py              # Reference implementation (Model)
├── custom_code.py               # Example optimized implementation (ModelNew)
└── run_iterx.py                 # IterX evaluation runner
```

---

## Running get_reward

```bash
python eval_cuda.py
```

```bash
python eval_cuda_server.py
```

---

## Running iterX

```bash
python run_iterx.py
```


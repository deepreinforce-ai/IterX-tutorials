import torch
import triton
import triton.language as tl


@triton.jit
def _h100_fp8_gemm_kernel(
    A_ptr,
    W_fp8_ptr,
    W_scale_ptr,
    Out_ptr,
    expert_ids_ptr,
    M, K, N,
    num_experts,
    stride_am, stride_ak,
    stride_we, stride_wn, stride_wk,
    stride_wse, stride_wsn, stride_wsk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    """
    H100-optimized FP8 GEMM with inline block-scale dequantization.
    
    Tile configuration optimized for irregular MoE workloads:
    - BLOCK_M=1: Token-level granularity for correct expert routing
    - BLOCK_N=128: H100 tensor core optimal width for FP8
    - BLOCK_K=128: Maximized arithmetic intensity for tensor cores
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_m >= M:
        return
    
    expert_id = tl.load(expert_ids_ptr + pid_m)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        w_fp8_ptrs = W_fp8_ptr + (expert_id * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_fp8 = tl.load(w_fp8_ptrs, mask=w_mask, other=0.0)
        
        scale_n_idx = offs_n // SCALE_BLOCK
        scale_k_idx = k // SCALE_BLOCK
        w_scale_ptrs = W_scale_ptr + (expert_id * stride_wse + scale_n_idx * stride_wsn + scale_k_idx * stride_wsk)
        w_scale_mask = (scale_n_idx < tl.cdiv(N, SCALE_BLOCK)) & (scale_k_idx < tl.cdiv(K, SCALE_BLOCK))
        w_scales = tl.load(w_scale_ptrs, mask=w_scale_mask, other=1.0)
        
        w_fp32 = w_fp8.to(tl.float32)
        w_dequant = w_fp32 * w_scales[:, None]
        
        acc += tl.dot(a, tl.trans(w_dequant))
    
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _optimal_swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Bandwidth-optimized SwiGLU kernel: gate * sigmoid(gate) * up.
    
    BLOCK_SIZE=16384 provides optimal balance for H100:
    - High memory throughput utilization
    - Minimal kernel launch overhead
    - Maintains good SM occupancy
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    sigmoid_gate = tl.sigmoid(gate)
    result = gate * sigmoid_gate * up
    
    tl.store(out_ptr + offs, result, mask=mask)


def vectorized_hidden_dequant(hidden_states, scales, block_size=128):
    """
    Vectorized FP8 dequantization using PyTorch's optimized kernels.
    Performed once and reused across all experts to amortize cost.
    """
    T, H = hidden_states.shape
    hidden_fp32 = hidden_states.to(torch.float32)
    
    scales_t = scales.t().contiguous()
    scales_expanded = scales_t.repeat_interleave(block_size, dim=1)
    
    return hidden_fp32 * scales_expanded


def fully_vectorized_token_expert_mapping(topk_idx, weights, local_expert_start, E_local, E_global):
    """
    Fully vectorized token-expert assignment using GPU-parallel operations.
    Eliminates Python loops for maximum performance.
    
    Args:
        topk_idx: [T, TOP_K] int32 selected expert indices
        weights: [T, E_global] float32 routing weights
        local_expert_start: int offset of local experts
        E_local: int number of local experts
        E_global: int total number of experts
        
    Returns:
        all_tokens: [M] int64 token indices
        all_experts: [M] int64 local expert indices
        all_weights: [M] float32 routing weights
    """
    T, TOP_K = topk_idx.shape
    device = topk_idx.device
    
    local_end = local_expert_start + E_local
    
    valid_mask = (topk_idx >= local_expert_start) & (topk_idx < local_end)
    
    if not valid_mask.any():
        return None, None, None
    
    token_indices, k_positions = torch.where(valid_mask)
    
    global_expert_ids = topk_idx[token_indices, k_positions]
    local_expert_ids = global_expert_ids - local_expert_start
    
    expert_weights = weights[token_indices, global_expert_ids]
    
    return token_indices, local_expert_ids, expert_weights


def batched_fp8_gemm(A, W_fp8, W_scale, expert_ids, scale_block=128):
    """
    Batched FP8 GEMM with H100-optimized tile configuration.
    
    Args:
        A: [M, K] FP32 activations
        W_fp8: [E, N, K] FP8 weights
        W_scale: [E, N//scale_block, K//scale_block] FP32 scales
        expert_ids: [M] int32 expert indices
        
    Returns:
        [M, N] FP32 output
    """
    M, K = A.shape
    E, N, _ = W_fp8.shape
    
    Out = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    BLOCK_M = 1
    BLOCK_N = 128
    BLOCK_K = 128
    
    grid = (M, triton.cdiv(N, BLOCK_N))
    
    _h100_fp8_gemm_kernel[grid](
        A, W_fp8, W_scale, Out,
        expert_ids,
        M, K, N, E,
        A.stride(0), A.stride(1),
        W_fp8.stride(0), W_fp8.stride(1), W_fp8.stride(2),
        W_scale.stride(0), W_scale.stride(1), W_scale.stride(2),
        Out.stride(0), Out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SCALE_BLOCK=scale_block,
    )
    
    return Out


def optimal_swiglu(gate, up):
    """
    Bandwidth-optimized SwiGLU activation function.
    
    Args:
        gate: [N] FP32 tensor
        up: [N] FP32 tensor
        
    Returns:
        [N] FP32 tensor (gate * sigmoid(gate) * up)
    """
    N = gate.numel()
    out = torch.empty_like(gate, dtype=torch.float32)
    
    BLOCK_SIZE = 16384
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    _optimal_swiglu_kernel[grid](
        gate.contiguous(),
        up.contiguous(),
        out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    """
    H100-optimized MoE inference with fully vectorized token-expert assignment.
    
    Pipeline:
    1. PyTorch-native group-based top-k routing
    2. Fully vectorized token-expert mapping (GPU-parallel)
    3. Vectorized activation dequantization (amortized)
    4. Batched GEMM1 with inline FP8 dequantization
    5. Bandwidth-optimized SwiGLU activation
    6. Batched GEMM2 with inline FP8 dequantization
    7. Weighted accumulation
    """
    original_device = hidden_states.device
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        routing_logits = routing_logits.cuda() if routing_logits.device.type != 'cuda' else routing_logits
        routing_bias = routing_bias.cuda() if routing_bias.device.type != 'cuda' else routing_bias
        hidden_states = hidden_states.cuda() if hidden_states.device.type != 'cuda' else hidden_states
        hidden_states_scale = hidden_states_scale.cuda() if hidden_states_scale.device.type != 'cuda' else hidden_states_scale
        gemm1_weights = gemm1_weights.cuda() if gemm1_weights.device.type != 'cuda' else gemm1_weights
        gemm1_weights_scale = gemm1_weights_scale.cuda() if gemm1_weights_scale.device.type != 'cuda' else gemm1_weights_scale
        gemm2_weights = gemm2_weights.cuda() if gemm2_weights.device.type != 'cuda' else gemm2_weights
        gemm2_weights_scale = gemm2_weights_scale.cuda() if gemm2_weights_scale.device.type != 'cuda' else gemm2_weights_scale
    else:
        if any(t.device.type == 'cuda' for t in [routing_logits, routing_bias, hidden_states, hidden_states_scale,
                                                   gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale]):
            raise RuntimeError("CUDA tensors provided but CUDA is not available")
        device = hidden_states.device
    
    H = 7168
    I = 2048
    BLOCK = 128
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    
    s = torch.sigmoid(logits)
    s_with_bias = s + bias
    
    group_size = E_global // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros(T, N_GROUP, device=device, dtype=torch.float32)
    group_mask.scatter_(1, group_idx, 1.0)
    
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)
    
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)
    
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor
    
    local_start = int(local_expert_offset)
    
    all_tokens, all_experts, all_weights = fully_vectorized_token_expert_mapping(
        topk_idx, weights, local_start, E_local, E_global
    )
    
    if all_tokens is None:
        result = torch.zeros(T, H, dtype=torch.bfloat16, device=device)
        if original_device.type != 'cuda' and result.device.type == 'cuda':
            result = result.cpu()
        return result
    
    A_all = vectorized_hidden_dequant(hidden_states, hidden_states_scale, BLOCK)
    A_batch = A_all[all_tokens]
    
    G1 = batched_fp8_gemm(
        A_batch,
        gemm1_weights,
        gemm1_weights_scale,
        all_experts,
        scale_block=BLOCK
    )
    
    gate = G1[:, I:].contiguous()
    up = G1[:, :I].contiguous()
    C = optimal_swiglu(gate, up)
    
    O = batched_fp8_gemm(
        C,
        gemm2_weights,
        gemm2_weights_scale,
        all_experts,
        scale_block=BLOCK
    )
    
    output = torch.zeros(T, H, dtype=torch.float32, device=device)
    O_weighted = O * all_weights.unsqueeze(1)
    output.index_add_(0, all_tokens, O_weighted)
    
    result = output.to(torch.bfloat16)
    
    if original_device.type != 'cuda' and result.device.type == 'cuda':
        result = result.cpu()
    
    return result
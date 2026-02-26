import torch
import triton
import triton.language as tl


@triton.jit
def dequant_hidden_kernel(
    H_fp8_ptr, H_scale_ptr, H_out_ptr,
    T, H_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    stride_h_t, stride_h_h,
    stride_s_b, stride_s_t,
    stride_o_t, stride_o_h,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_t = offs_t < T
    mask_h = offs_h < H_dim
    mask = mask_t[:, None] & mask_h[None, :]

    h_ptrs = H_fp8_ptr + offs_t[:, None] * stride_h_t + offs_h[None, :] * stride_h_h
    h_val = tl.load(h_ptrs, mask=mask, other=0.0).to(tl.float32)

    h_block = offs_h // BLOCK_SIZE
    s_ptrs = H_scale_ptr + h_block[None, :] * stride_s_b + offs_t[:, None] * stride_s_t
    s_val = tl.load(s_ptrs, mask=mask, other=1.0)

    out_ptrs = H_out_ptr + offs_t[:, None] * stride_o_t + offs_h[None, :] * stride_o_h
    tl.store(out_ptrs, h_val * s_val, mask=mask)


@triton.jit
def dequant_weight_kernel(
    W_fp8_ptr, W_scale_ptr, W_out_ptr,
    N, K,
    BLOCK_SIZE: tl.constexpr,
    stride_w_n, stride_w_k,
    stride_s_n, stride_s_k,
    stride_o_n, stride_o_k,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_k = offs_k < K
    mask = mask_n[:, None] & mask_k[None, :]

    w_ptrs = W_fp8_ptr + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    n_block = offs_n // BLOCK_SIZE
    k_block = offs_k // BLOCK_SIZE
    s_ptrs = W_scale_ptr + n_block[:, None] * stride_s_n + k_block[None, :] * stride_s_k
    s = tl.load(s_ptrs, mask=mask, other=1.0)

    out_ptrs = W_out_ptr + offs_n[:, None] * stride_o_n + offs_k[None, :] * stride_o_k
    tl.store(out_ptrs, w * s, mask=mask)


@triton.jit
def fused_swiglu_kernel(
    G1_ptr, Out_ptr,
    Tk, I_dim: tl.constexpr,
    stride_g_t, stride_g_i,
    stride_o_t, stride_o_i,
    BLOCK_T: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_t = offs_t < Tk
    mask_i = offs_i < I_dim
    mask = mask_t[:, None] & mask_i[None, :]

    x1_ptrs = G1_ptr + offs_t[:, None] * stride_g_t + offs_i[None, :] * stride_g_i
    x1 = tl.load(x1_ptrs, mask=mask, other=0.0)

    x2_ptrs = G1_ptr + offs_t[:, None] * stride_g_t + (offs_i[None, :] + I_dim) * stride_g_i
    x2 = tl.load(x2_ptrs, mask=mask, other=0.0)

    silu_x2 = x2 * tl.sigmoid(x2)
    result = silu_x2 * x1

    out_ptrs = Out_ptr + offs_t[:, None] * stride_o_t + offs_i[None, :] * stride_o_i
    tl.store(out_ptrs, result, mask=mask)


@triton.jit
def weighted_scatter_kernel(
    O_ptr, W_ptr, Token_ids_ptr, Out_ptr,
    Tk, H_dim: tl.constexpr,
    stride_o_t, stride_o_h,
    stride_out_t, stride_out_h,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= Tk:
        return

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H_dim

    token_id = tl.load(Token_ids_ptr + pid_t)
    w = tl.load(W_ptr + pid_t)

    o_ptrs = O_ptr + pid_t * stride_o_t + offs_h * stride_o_h
    o_val = tl.load(o_ptrs, mask=mask_h, other=0.0)

    weighted = o_val * w

    out_ptrs = Out_ptr + token_id * stride_out_t + offs_h * stride_out_h
    tl.atomic_add(out_ptrs, weighted, mask=mask_h)


@triton.jit
def routing_sigmoid_bias_kernel(
    logits_ptr, bias_ptr, s_ptr, s_wb_ptr,
    T, E: tl.constexpr,
    stride_l_t, stride_l_e,
    stride_s_t, stride_s_e,
    stride_sw_t, stride_sw_e,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_e = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    mask_t = offs_t < T
    mask_e = offs_e < E
    mask = mask_t[:, None] & mask_e[None, :]

    l_ptrs = logits_ptr + offs_t[:, None] * stride_l_t + offs_e[None, :] * stride_l_e
    logits = tl.load(l_ptrs, mask=mask, other=0.0).to(tl.float32)

    b_ptrs = bias_ptr + offs_e
    bias = tl.load(b_ptrs, mask=mask_e, other=0.0).to(tl.float32)

    sig = tl.sigmoid(logits)

    s_out = s_ptr + offs_t[:, None] * stride_s_t + offs_e[None, :] * stride_s_e
    tl.store(s_out, sig, mask=mask)

    s_wb = sig + bias[None, :]
    sw_out = s_wb_ptr + offs_t[:, None] * stride_sw_t + offs_e[None, :] * stride_sw_e
    tl.store(sw_out, s_wb, mask=mask)


def _dequant_w13(gemm1_weights, gemm1_weights_scale, le_idx, buf, I, H, BLOCK, BN, BK):
    w_fp8 = gemm1_weights[le_idx]
    w_s = gemm1_weights_scale[le_idx]
    grid = (triton.cdiv(2 * I, BN), triton.cdiv(H, BK))
    dequant_weight_kernel[grid](
        w_fp8, w_s, buf,
        2 * I, H, BLOCK,
        w_fp8.stride(0), w_fp8.stride(1),
        w_s.stride(0), w_s.stride(1),
        buf.stride(0), buf.stride(1),
        BLOCK_N=BN, BLOCK_K=BK,
    )


def _dequant_w2(gemm2_weights, gemm2_weights_scale, le_idx, buf, H, I, BLOCK, BN, BK):
    w_fp8 = gemm2_weights[le_idx]
    w_s = gemm2_weights_scale[le_idx]
    grid = (triton.cdiv(H, BN), triton.cdiv(I, BK))
    dequant_weight_kernel[grid](
        w_fp8, w_s, buf,
        H, I, BLOCK,
        w_fp8.stride(0), w_fp8.stride(1),
        w_s.stride(0), w_s.stride(1),
        buf.stride(0), buf.stride(1),
        BLOCK_N=BN, BLOCK_K=BK,
    )


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
    orig_device = hidden_states.device
    if orig_device.type != 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        routing_logits = routing_logits.cuda()
        routing_bias = routing_bias.cuda()
        hidden_states = hidden_states.cuda()
        hidden_states_scale = hidden_states_scale.cuda()
        gemm1_weights = gemm1_weights.cuda()
        gemm1_weights_scale = gemm1_weights_scale.cuda()
        gemm2_weights = gemm2_weights.cuda()
        gemm2_weights_scale = gemm2_weights_scale.cuda()

    device = hidden_states.device
    H = 7168
    I = 2048
    BLOCK = 128
    E_global = 256
    E_local = 32
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    T = routing_logits.shape[0]

    # ============ ROUTING: Triton sigmoid+bias, then PyTorch topk ============
    s = torch.empty((T, E_global), dtype=torch.float32, device=device)
    s_with_bias = torch.empty((T, E_global), dtype=torch.float32, device=device)

    BT_r, BE_r = 64, 256
    grid_r = (triton.cdiv(T, BT_r), triton.cdiv(E_global, BE_r))
    routing_sigmoid_bias_kernel[grid_r](
        routing_logits, routing_bias, s, s_with_bias,
        T, E_global,
        routing_logits.stride(0), routing_logits.stride(1),
        s.stride(0), s.stride(1),
        s_with_bias.stride(0), s_with_bias.stride(1),
        BLOCK_T=BT_r, BLOCK_E=BE_r,
    )

    group_size = E_global // N_GROUP  # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M_mask = torch.zeros_like(s)
    M_mask.scatter_(1, topk_idx, 1.0)
    weights = s * M_mask
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # ============ DEQUANTIZE HIDDEN STATES (Triton) ============
    A = torch.empty((T, H), dtype=torch.float32, device=device)
    BT_h, BH_h = 32, 128
    grid_h = (triton.cdiv(T, BT_h), triton.cdiv(H, BH_h))
    dequant_hidden_kernel[grid_h](
        hidden_states, hidden_states_scale, A,
        T, H, BLOCK,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        A.stride(0), A.stride(1),
        BLOCK_T=BT_h, BLOCK_H=BH_h,
    )

    # ============ BUILD EXPERT-TOKEN MAPPING (vectorized) ============
    local_start = int(local_expert_offset)

    flat_experts = topk_idx.reshape(-1)
    flat_token_ids = torch.arange(T, device=device, dtype=torch.long).unsqueeze(1).expand(T, TOP_K).reshape(-1)

    local_mask = (flat_experts >= local_start) & (flat_experts < local_start + E_local)
    local_flat_le = flat_experts[local_mask] - local_start
    local_flat_tokens = flat_token_ids[local_mask]

    sorted_indices = torch.argsort(local_flat_le, stable=True)
    sorted_le = local_flat_le[sorted_indices]
    sorted_tokens = local_flat_tokens[sorted_indices]

    expert_counts = torch.zeros(E_local, dtype=torch.long, device=device)
    if sorted_le.numel() > 0:
        expert_counts.scatter_add_(0, sorted_le.long(), torch.ones_like(sorted_le, dtype=torch.long))
    expert_offsets_arr = torch.zeros(E_local + 1, dtype=torch.long, device=device)
    expert_offsets_arr[1:] = torch.cumsum(expert_counts, dim=0)

    expert_offsets_cpu = expert_offsets_arr.cpu()
    active_experts = []
    active_token_indices = []
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue
        start_off = expert_offsets_cpu[le].item()
        end_off = expert_offsets_cpu[le + 1].item()
        if start_off == end_off:
            continue
        token_idx = torch.unique(sorted_tokens[start_off:end_off])
        if token_idx.numel() == 0:
            continue
        active_experts.append(le)
        active_token_indices.append(token_idx)

    # ============ EXPERT COMPUTATION with 3-stage pipelining ============
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    NUM_BUF = 3
    W_buf_13 = [torch.empty((2 * I, H), dtype=torch.float32, device=device) for _ in range(NUM_BUF)]
    W_buf_2 = [torch.empty((H, I), dtype=torch.float32, device=device) for _ in range(NUM_BUF)]
    # Pre-allocate a reusable SwiGLU buffer
    max_tokens = max(T, 1)
    swiglu_buf = torch.empty((max_tokens, I), dtype=torch.float32, device=device)

    BN_w, BK_w = 128, 128
    num_active = len(active_experts)

    # Prefetch W13 and W2 for first 2 experts
    for prefetch_idx in range(min(2, num_active)):
        le_p = active_experts[prefetch_idx]
        _dequant_w13(gemm1_weights, gemm1_weights_scale, le_p,
                     W_buf_13[prefetch_idx], I, H, BLOCK, BN_w, BK_w)
        _dequant_w2(gemm2_weights, gemm2_weights_scale, le_p,
                    W_buf_2[prefetch_idx], H, I, BLOCK, BN_w, BK_w)

    for idx in range(num_active):
        le = active_experts[idx]
        ge = local_start + le
        token_idx = active_token_indices[idx]
        Tk = token_idx.numel()

        buf_idx = idx % NUM_BUF
        cur_w13 = W_buf_13[buf_idx]
        cur_w2 = W_buf_2[buf_idx]

        A_e = A.index_select(0, token_idx)

        # GEMM1: [Tk, H] @ [H, 2I] = [Tk, 2I] via cuBLAS
        G1 = torch.mm(A_e, cur_w13.t())

        # Pipeline: prefetch W13 and W2 for idx+2
        if idx + 2 < num_active:
            next_le = active_experts[idx + 2]
            next_buf = (idx + 2) % NUM_BUF
            _dequant_w13(gemm1_weights, gemm1_weights_scale, next_le,
                         W_buf_13[next_buf], I, H, BLOCK, BN_w, BK_w)
            _dequant_w2(gemm2_weights, gemm2_weights_scale, next_le,
                        W_buf_2[next_buf], H, I, BLOCK, BN_w, BK_w)

        # SwiGLU (Triton fused) - reuse buffer
        C = swiglu_buf[:Tk]
        grid_sg = (triton.cdiv(Tk, 64), triton.cdiv(I, 256))
        fused_swiglu_kernel[grid_sg](
            G1, C, Tk, I,
            G1.stride(0), G1.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_T=64, BLOCK_I=256,
        )

        # GEMM2: [Tk, I] @ [I, H] = [Tk, H] via cuBLAS
        O = torch.mm(C, cur_w2.t())

        # Weighted scatter add - one program per token
        w_tok = weights[token_idx, ge].contiguous()
        BH_wa = 256
        grid_wa = (Tk, triton.cdiv(H, BH_wa))
        weighted_scatter_kernel[grid_wa](
            O, w_tok, token_idx, output,
            Tk, H,
            O.stride(0), O.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_H=BH_wa,
        )

    result = output.to(torch.bfloat16)
    if orig_device.type != 'cuda':
        result = result.to(orig_device)
    return result
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_gemm1_fp8_kernel(
    A_ptr, stride_am, stride_ak,
    AS_ptr, stride_ast, stride_asb,
    W_ptr, stride_we, stride_wn, stride_wk,
    WS_ptr, stride_wse, stride_wsn, stride_wsk,
    Out_ptr, stride_om, stride_on,
    ExpertBnd_ptr,
    M_total, K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)

    e_start = tl.load(ExpertBnd_ptr + pid_e)
    e_end = tl.load(ExpertBnd_ptr + pid_e + 1)

    if e_end <= e_start:
        return

    tile_m_start = e_start + pid_m * BLOCK_M
    if tile_m_start >= e_end:
        return

    offs_m = tile_m_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < e_end
    n_mask = offs_n < N

    sn_idx = offs_n // 128

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        sk = k_start // 128

        a_fp8 = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=m_mask[:, None], other=0.0
        )

        w_fp8 = tl.load(
            W_ptr + pid_e * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=n_mask[:, None], other=0.0
        )

        # Native FP8 tensor cores
        partial = tl.dot(a_fp8.to(tl.float8e4nv), tl.trans(w_fp8.to(tl.float8e4nv)), out_dtype=tl.float32)

        # A block scales [BLOCK_M]
        a_scale = tl.load(
            AS_ptr + offs_m * stride_ast + sk * stride_asb,
            mask=m_mask, other=1.0
        ).to(tl.float32)

        # W block scales [BLOCK_N]
        w_scale = tl.load(
            WS_ptr + pid_e * stride_wse + sn_idx * stride_wsn + sk * stride_wsk,
            mask=n_mask, other=1.0
        ).to(tl.float32)

        acc = acc + partial * (a_scale[:, None] * w_scale[None, :])

    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc, mask=out_mask
    )


@triton.jit
def _swiglu_fp8quant_kernel(
    G1_ptr,
    Out_fp8_ptr,
    Out_scale_ptr,  # [M_total, I//128] output scales
    M, I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """SwiGLU + quantize output to FP8 for GEMM2."""
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (offs_m[:, None] < M) & (offs_i[None, :] < I)

    up = tl.load(G1_ptr + offs_m[:, None] * (2 * I) + offs_i[None, :],
                 mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(G1_ptr + offs_m[:, None] * (2 * I) + I + offs_i[None, :],
                   mask=mask, other=0.0).to(tl.float32)

    result = gate * tl.sigmoid(gate) * up  # [BLOCK_M, BLOCK_I]

    # Store as float32 (FP8 quantization done separately for simplicity)
    tl.store(Out_fp8_ptr + offs_m[:, None] * I + offs_i[None, :], result, mask=mask)


@triton.jit
def _moe_gemm2_fp8_kernel(
    A_ptr, stride_am, stride_ak,      # [M_total, K] float32
    AS_ptr, stride_ast, stride_asb,   # [M_total, K//128] float32 scales
    W_ptr, stride_we, stride_wn, stride_wk,
    WS_ptr, stride_wse, stride_wsn, stride_wsk,
    Out_ptr, stride_om, stride_on,
    ExpertBnd_ptr,
    M_total, K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)

    e_start = tl.load(ExpertBnd_ptr + pid_e)
    e_end = tl.load(ExpertBnd_ptr + pid_e + 1)

    if e_end <= e_start:
        return

    tile_m_start = e_start + pid_m * BLOCK_M
    if tile_m_start >= e_end:
        return

    offs_m = tile_m_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < e_end
    n_mask = offs_n < N

    sn_idx = offs_n // 128
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        sk = k_start // 128

        # Load A float32
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=m_mask[:, None], other=0.0
        ).to(tl.float32)

        # Load W FP8
        w_fp8 = tl.load(
            W_ptr + pid_e * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=n_mask[:, None], other=0.0
        )

        # Load A scales [BLOCK_M]
        a_scale = tl.load(
            AS_ptr + offs_m * stride_ast + sk * stride_asb,
            mask=m_mask, other=1.0
        ).to(tl.float32)

        # Load W scales [BLOCK_N]
        w_scale = tl.load(
            WS_ptr + pid_e * stride_wse + sn_idx * stride_wsn + sk * stride_wsk,
            mask=n_mask, other=1.0
        ).to(tl.float32)

        # Quantize A block to FP8 for tensor cores
        # Find per-block max for quantization
        a_abs_max = tl.max(tl.abs(a))
        fp8_max = 448.0  # max for float8_e4m3fn
        q_scale = a_abs_max / fp8_max + 1e-12
        a_fp8 = (a / q_scale).to(tl.float8e4nv)

        w_f32_scaled = w_fp8.to(tl.float32)
        # Re-encode W as FP8 (it already is, just cast)
        w_fp8_cast = w_f32_scaled.to(tl.float8e4nv)

        partial = tl.dot(a_fp8, tl.trans(w_fp8_cast), out_dtype=tl.float32)

        combined_scale = q_scale * a_scale[:, None] * w_scale[None, :]
        acc = acc + partial * combined_scale

    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc, mask=out_mask
    )


@triton.jit
def _moe_gemm2_f32_fp8_kernel(
    A_ptr, stride_am, stride_ak,      # [M_total, K] float32
    W_ptr, stride_we, stride_wn, stride_wk,
    WS_ptr, stride_wse, stride_wsn, stride_wsk,
    Out_ptr, stride_om, stride_on,
    ExpertBnd_ptr,
    M_total, K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """GEMM2: float32 A @ fp8 W (no FP8 tensor cores, using tf32)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)

    e_start = tl.load(ExpertBnd_ptr + pid_e)
    e_end = tl.load(ExpertBnd_ptr + pid_e + 1)

    if e_end <= e_start:
        return

    tile_m_start = e_start + pid_m * BLOCK_M
    if tile_m_start >= e_end:
        return

    offs_m = tile_m_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < e_end
    n_mask = offs_n < N

    sn_idx = offs_n // 128
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        sk = k_start // 128

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=m_mask[:, None], other=0.0
        ).to(tl.float32)

        w_fp8 = tl.load(
            W_ptr + pid_e * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=n_mask[:, None], other=0.0
        )
        w_f32 = w_fp8.to(tl.float32)

        w_scale = tl.load(
            WS_ptr + pid_e * stride_wse + sn_idx * stride_wsn + sk * stride_wsk,
            mask=n_mask, other=1.0
        ).to(tl.float32)

        partial = tl.dot(a, tl.trans(w_f32), out_dtype=tl.float32)
        acc = acc + partial * w_scale[None, :]

    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc, mask=out_mask
    )


@triton.jit
def _swiglu_kernel(
    G1_ptr,
    Out_ptr,
    M, I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (offs_m[:, None] < M) & (offs_i[None, :] < I)

    up = tl.load(G1_ptr + offs_m[:, None] * (2 * I) + offs_i[None, :],
                 mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(G1_ptr + offs_m[:, None] * (2 * I) + I + offs_i[None, :],
                   mask=mask, other=0.0).to(tl.float32)

    result = gate * tl.sigmoid(gate) * up
    tl.store(Out_ptr + offs_m[:, None] * I + offs_i[None, :], result, mask=mask)


@triton.jit
def _weighted_scatter_kernel(
    G2_ptr, stride_gm, stride_gn,         # [M_total, H]
    Weights_ptr,                            # [M_total]
    TokIdx_ptr,                             # [M_total]
    Out_ptr, stride_om, stride_on,          # [T, H]
    M_total, H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Apply weights and scatter-add to output."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    tok_id = tl.load(TokIdx_ptr + pid_m).to(tl.int64)
    w = tl.load(Weights_ptr + pid_m).to(tl.float32)

    g = tl.load(G2_ptr + pid_m * stride_gm + offs_h * stride_gn, mask=h_mask, other=0.0).to(tl.float32)
    weighted = g * w

    # atomic add to output
    tl.atomic_add(Out_ptr + tok_id * stride_om + offs_h * stride_on, weighted, mask=h_mask)


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
    original_device = hidden_states.device

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    def to_cuda(t):
        return t.cuda() if t.device.type != 'cuda' else t

    routing_logits = to_cuda(routing_logits)
    routing_bias = to_cuda(routing_bias)
    hidden_states = to_cuda(hidden_states)
    hidden_states_scale = to_cuda(hidden_states_scale)
    gemm1_weights = to_cuda(gemm1_weights)
    gemm1_weights_scale = to_cuda(gemm1_weights_scale)
    gemm2_weights = to_cuda(gemm2_weights)
    gemm2_weights_scale = to_cuda(gemm2_weights_scale)

    device = torch.device('cuda')

    H = 7168
    I_size = 2048
    BLOCK = 128
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    local_start = int(local_expert_offset)

    num_hidden_blocks = H // BLOCK      # 56
    num_inter_blocks = I_size // BLOCK  # 16

    # === Routing (PyTorch) ===
    logits = routing_logits.float()
    bias = routing_bias.float().reshape(-1)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    group_size = E_global // N_GROUP  # 32
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

    M_mask = torch.zeros(T, E_global, device=device, dtype=torch.float32)
    M_mask.scatter_(1, topk_idx, 1.0)
    weights_unnorm = s * M_mask
    weights_sum = weights_unnorm.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights_unnorm / weights_sum) * routed_scaling_factor

    # === Token-expert mapping ===
    local_end = local_start + E_local
    valid_mask = (topk_idx >= local_start) & (topk_idx < local_end)

    if not valid_mask.any():
        result = torch.zeros(T, H, dtype=torch.bfloat16, device=device)
        if original_device.type != 'cuda':
            result = result.to(original_device)
        return result

    token_indices, k_positions = torch.where(valid_mask)
    global_expert_ids = topk_idx[token_indices, k_positions]
    local_expert_ids = (global_expert_ids - local_start).to(torch.int64)
    expert_weights_vals = weights[token_indices, global_expert_ids].float()

    # Sort by local expert id for grouped GEMM
    sort_order = torch.argsort(local_expert_ids, stable=True)
    token_indices_sorted = token_indices[sort_order].to(torch.int64)
    local_expert_ids_sorted = local_expert_ids[sort_order]
    expert_weights_sorted = expert_weights_vals[sort_order]

    M_total = int(token_indices_sorted.shape[0])

    # Expert boundaries
    expert_range = torch.arange(E_local + 1, dtype=torch.int64, device=device)
    expert_boundaries = torch.searchsorted(local_expert_ids_sorted.to(torch.int64), expert_range)

    # Gather FP8 hidden states [M_total, H]
    hidden_states_c = hidden_states.contiguous()
    A_gathered = hidden_states_c[token_indices_sorted].contiguous()

    # Pre-transpose A scales: [H//128, T] -> [T, H//128]
    AS_t = hidden_states_scale.t().contiguous()  # [T, 56]
    # Gather scales for selected tokens [M_total, 56]
    A_scales_gathered = AS_t[token_indices_sorted].contiguous()

    # === GEMM1: FP8 × FP8 -> float32, [M_total, 4096] ===
    G1_out = torch.empty(M_total, 2 * I_size, dtype=torch.float32, device=device)

    BLOCK_M1 = 32
    BLOCK_N1 = 256
    BLOCK_K1 = 128

    max_tiles_m1 = triton.cdiv(M_total, BLOCK_M1) + 1
    grid1 = (max_tiles_m1, triton.cdiv(2 * I_size, BLOCK_N1), E_local)

    _moe_gemm1_fp8_kernel[grid1](
        A_gathered, A_gathered.stride(0), A_gathered.stride(1),
        A_scales_gathered, A_scales_gathered.stride(0), A_scales_gathered.stride(1),
        gemm1_weights, gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_weights_scale, gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        G1_out, G1_out.stride(0), G1_out.stride(1),
        expert_boundaries,
        M_total, H, 2 * I_size,
        BLOCK_M=BLOCK_M1,
        BLOCK_N=BLOCK_N1,
        BLOCK_K=BLOCK_K1,
    )

    # === SwiGLU: [M_total, 4096] -> [M_total, 2048] ===
    C = torch.empty(M_total, I_size, dtype=torch.float32, device=device)

    BLOCK_M_SWI = 32
    BLOCK_I_SWI = 512
    swiglu_grid = (triton.cdiv(M_total, BLOCK_M_SWI), triton.cdiv(I_size, BLOCK_I_SWI))
    _swiglu_kernel[swiglu_grid](
        G1_out, C,
        M_total, I_size,
        BLOCK_M=BLOCK_M_SWI,
        BLOCK_I=BLOCK_I_SWI,
    )
    del G1_out

    # === GEMM2: float32 @ FP8 -> float32, [M_total, 7168] ===
    G2_out = torch.empty(M_total, H, dtype=torch.float32, device=device)

    BLOCK_M2 = 32
    BLOCK_N2 = 256
    BLOCK_K2 = 128

    max_tiles_m2 = triton.cdiv(M_total, BLOCK_M2) + 1
    grid2 = (max_tiles_m2, triton.cdiv(H, BLOCK_N2), E_local)

    _moe_gemm2_f32_fp8_kernel[grid2](
        C, C.stride(0), C.stride(1),
        gemm2_weights, gemm2_weights.stride(0), gemm2_weights.stride(1), gemm2_weights.stride(2),
        gemm2_weights_scale, gemm2_weights_scale.stride(0), gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
        G2_out, G2_out.stride(0), G2_out.stride(1),
        expert_boundaries,
        M_total, I_size, H,
        BLOCK_M=BLOCK_M2,
        BLOCK_N=BLOCK_N2,
        BLOCK_K=BLOCK_K2,
    )
    del C

    # === Weighted accumulation ===
    output = torch.zeros(T, H, dtype=torch.float32, device=device)
    if M_total > 0:
        G2_weighted = G2_out * expert_weights_sorted.unsqueeze(1)
        output.index_add_(0, token_indices_sorted, G2_weighted)
    del G2_out

    result = output.to(torch.bfloat16)

    if original_device.type != 'cuda':
        result = result.to(original_device)

    return result
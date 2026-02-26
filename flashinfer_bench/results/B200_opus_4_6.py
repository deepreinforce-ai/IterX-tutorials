import torch
import triton
import triton.language as tl


@triton.jit
def fused_gemm1_blockscale_kernel(
    A_fp8_ptr, A_scale_ptr,
    W_fp8_ptr, W_scale_ptr,
    token_idx_ptr, out_ptr,
    T, Tk,
    H: tl.constexpr, N: tl.constexpr,
    QUANT_BLOCK: tl.constexpr, H_blocks: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = N // BLOCK_N
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_off < Tk

    token_ids = tl.load(token_idx_ptr + m_off, mask=m_mask, other=0)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    n_block_idx = n_off // QUANT_BLOCK

    for k_start in range(0, H, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_block_idx = k_start // QUANT_BLOCK

        a_fp8 = tl.load(A_fp8_ptr + token_ids[:, None] * H + k_off[None, :],
                        mask=m_mask[:, None], other=0.0).to(tl.float32)
        a_scale = tl.load(A_scale_ptr + k_block_idx * T + token_ids,
                         mask=m_mask, other=1.0)
        a_vals = a_fp8 * a_scale[:, None]

        w_fp8 = tl.load(W_fp8_ptr + n_off[:, None] * H + k_off[None, :]).to(tl.float32)
        w_scale = tl.load(W_scale_ptr + n_block_idx * H_blocks + k_block_idx)
        w_vals = w_fp8 * w_scale[:, None]

        acc += tl.dot(a_vals, tl.trans(w_vals))

    tl.store(out_ptr + m_off[:, None] * N + n_off[None, :], acc, mask=m_mask[:, None])


@triton.jit
def fused_gemm2_weighted_scatter_kernel(
    C_ptr, W2_fp8_ptr, W2_scale_ptr,
    token_idx_ptr, w_tok_ptr, output_ptr,
    Tk,
    H: tl.constexpr, I_size: tl.constexpr,
    QUANT_BLOCK: tl.constexpr, H_blocks: tl.constexpr, I_blocks: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = H // BLOCK_N
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_off < Tk

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    n_block_idx = n_off // QUANT_BLOCK

    for k_start in range(0, I_size, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_block_idx = k_start // QUANT_BLOCK

        c_vals = tl.load(C_ptr + m_off[:, None] * I_size + k_off[None, :],
                        mask=m_mask[:, None], other=0.0)

        w_fp8 = tl.load(W2_fp8_ptr + n_off[:, None] * I_size + k_off[None, :]).to(tl.float32)
        w_scale = tl.load(W2_scale_ptr + n_block_idx * I_blocks + k_block_idx)
        w_vals = w_fp8 * w_scale[:, None]

        acc += tl.dot(c_vals, tl.trans(w_vals))

    w = tl.load(w_tok_ptr + m_off, mask=m_mask, other=0.0)
    acc = acc * w[:, None]

    token_ids = tl.load(token_idx_ptr + m_off, mask=m_mask, other=0)
    tl.atomic_add(output_ptr + token_ids[:, None] * H + n_off[None, :], acc, mask=m_mask[:, None])


@triton.jit
def fused_silu_mul_kernel(
    input_ptr, output_ptr,
    num_rows, I: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    if pid_row >= num_rows:
        return

    col_off = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    col_mask = col_off < I

    up = tl.load(input_ptr + pid_row * 2 * I + col_off, mask=col_mask, other=0.0)
    gate = tl.load(input_ptr + pid_row * 2 * I + I + col_off, mask=col_mask, other=0.0)
    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up
    tl.store(output_ptr + pid_row * I + col_off, result, mask=col_mask)


@triton.jit
def dequant_weight_kernel(
    w_fp8_ptr, scale_ptr, out_ptr,
    N, K: tl.constexpr, QUANT_BLOCK: tl.constexpr, K_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = n_off < N
    vals = tl.load(w_fp8_ptr + n_off[:, None] * K + k_off[None, :], mask=n_mask[:, None], other=0.0).to(tl.float32)
    n_block = n_off // QUANT_BLOCK
    k_block = k_off // QUANT_BLOCK
    scale = tl.load(scale_ptr + n_block[:, None] * K_blocks + k_block[None, :], mask=n_mask[:, None], other=1.0)
    result = vals * scale
    tl.store(out_ptr + n_off[:, None] * K + k_off[None, :], result, mask=n_mask[:, None])


@triton.jit
def dequant_hidden_kernel(
    hidden_ptr, scale_ptr, output_ptr,
    T, H: tl.constexpr, QUANT_BLOCK: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    t_mask = t_off < T
    mask = t_mask[:, None]
    vals = tl.load(hidden_ptr + t_off[:, None] * H + h_off[None, :], mask=mask, other=0.0).to(tl.float32)
    block_idx = h_off // QUANT_BLOCK
    scale = tl.load(scale_ptr + block_idx[None, :] * T + t_off[:, None], mask=mask, other=1.0)
    result = vals * scale
    tl.store(output_ptr + t_off[:, None] * H + h_off[None, :], result, mask=mask)


@triton.jit
def weighted_scatter_add_kernel(
    expert_out_ptr, output_ptr, token_idx_ptr, weights_ptr,
    Tk, H: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= Tk:
        return
    h_off = pid_n * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H
    orig_token = tl.load(token_idx_ptr + pid_m)
    w = tl.load(weights_ptr + pid_m)
    vals = tl.load(expert_out_ptr + pid_m * H + h_off, mask=h_mask, other=0.0)
    vals = vals * w
    tl.atomic_add(output_ptr + orig_token * H + h_off, vals, mask=h_mask)


@triton.jit
def routing_sigmoid_bias_kernel(
    logits_ptr, bias_ptr, s_ptr, s_wb_ptr,
    T, E: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_e = tl.program_id(1)
    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    e_off = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    t_mask = t_off < T
    e_mask = e_off < E
    mask = t_mask[:, None] & e_mask[None, :]
    logits = tl.load(logits_ptr + t_off[:, None] * E + e_off[None, :], mask=mask, other=0.0)
    bias = tl.load(bias_ptr + e_off, mask=e_mask, other=0.0).to(tl.float32)
    s = tl.sigmoid(logits)
    s_wb = s + bias[None, :]
    tl.store(s_ptr + t_off[:, None] * E + e_off[None, :], s, mask=mask)
    tl.store(s_wb_ptr + t_off[:, None] * E + e_off[None, :], s_wb, mask=mask)


def _dequant_weight(w_fp8, scale, N, K, QUANT_BLOCK, device):
    K_blocks = K // QUANT_BLOCK
    out = torch.empty((N, K), dtype=torch.float32, device=device)
    BN = 128
    BK = 128
    grid = ((N + BN - 1) // BN, (K + BK - 1) // BK)
    dequant_weight_kernel[grid](w_fp8, scale, out, N, K, QUANT_BLOCK, K_blocks, BN, BK)
    return out


def _pick_block_m(Tk):
    if Tk <= 16:
        return 16
    elif Tk <= 32:
        return 32
    elif Tk <= 64:
        return 64
    else:
        return 128


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
    T = routing_logits.shape[0]
    E_global = 256
    E_local = 32
    H = 7168
    I = 2048
    BLOCK = 128
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    H_blocks = H // BLOCK  # 56
    I_blocks = I // BLOCK  # 16
    N_gemm1 = 2 * I  # 4096

    # ===== ROUTING with fused sigmoid+bias kernel =====
    logits = routing_logits.float()
    bias = routing_bias.float().reshape(-1)

    s = torch.empty((T, E_global), dtype=torch.float32, device=device)
    s_with_bias = torch.empty((T, E_global), dtype=torch.float32, device=device)

    if T > 0:
        BT_R = 64
        BE_R = 256
        grid_r = ((T + BT_R - 1) // BT_R, (E_global + BE_R - 1) // BE_R)
        routing_sigmoid_bias_kernel[grid_r](logits, bias, s, s_with_bias, T, E_global, BT_R, BE_R)

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

    del s, s_with_bias, s_wb_grouped, top2_vals, group_scores, group_mask, score_mask, scores_pruned, M_mask

    # ===== EXPERT->TOKEN MAPPING (vectorized) =====
    local_start = int(local_expert_offset)

    topk_idx_expanded = topk_idx.unsqueeze(2)  # [T, TOP_K, 1]
    local_expert_ids = torch.arange(local_start, local_start + E_local, device=device).view(1, 1, E_local)
    matches = (topk_idx_expanded == local_expert_ids).any(dim=1)  # [T, E_local]

    # Extract local weights slice once
    local_weights = weights[:, local_start:local_start + E_local].contiguous()

    expert_counts = matches.sum(dim=0)  # [E_local]

    expert_info = []
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue
        cnt = expert_counts[le].item()
        if cnt == 0:
            continue
        token_idx = torch.nonzero(matches[:, le], as_tuple=False).squeeze(1)
        expert_info.append((le, ge, token_idx, cnt))

    del topk_idx_expanded, local_expert_ids, matches, expert_counts, topk_idx, weights

    if not expert_info:
        output = torch.zeros((T, H), dtype=torch.bfloat16, device=device)
        if orig_device.type != 'cuda':
            output = output.to(orig_device)
        return output

    # Sort by descending token count
    expert_info.sort(key=lambda x: -x[3])

    # Threshold for fused kernels vs cuBLAS
    FUSED_THRESHOLD = 48

    needs_cublas = any(info[3] > FUSED_THRESHOLD for info in expert_info)

    # Lazily dequantize hidden states only for cuBLAS path
    A_dequant = None
    if needs_cublas and T > 0:
        A_dequant = torch.empty((T, H), dtype=torch.float32, device=device)
        BT = 32
        BH = 128
        grid_dq = ((T + BT - 1) // BT, (H + BH - 1) // BH)
        dequant_hidden_kernel[grid_dq](hidden_states, hidden_states_scale, A_dequant, T, H, BLOCK, BT, BH)

    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    # Pre-allocate reusable buffers
    max_tk = max(info[3] for info in expert_info)
    G1_buf = torch.empty((max_tk, N_gemm1), dtype=torch.float32, device=device)
    C_buf = torch.empty((max_tk, I), dtype=torch.float32, device=device)
    O_buf = torch.empty((max_tk, H), dtype=torch.float32, device=device) if needs_cublas else None

    for le, ge, token_idx, Tk in expert_info:
        if Tk == 0:
            continue

        use_fused = (Tk <= FUSED_THRESHOLD)
        BLOCK_M = _pick_block_m(Tk)

        # ===== GEMM1 =====
        G1 = G1_buf[:Tk]
        if use_fused:
            BLOCK_N1 = 128
            BLOCK_K1 = 128
            num_m = (Tk + BLOCK_M - 1) // BLOCK_M
            num_n = N_gemm1 // BLOCK_N1
            fused_gemm1_blockscale_kernel[(num_m * num_n,)](
                hidden_states, hidden_states_scale,
                gemm1_weights[le], gemm1_weights_scale[le],
                token_idx, G1,
                T, Tk, H, N_gemm1, BLOCK, H_blocks,
                BLOCK_M, BLOCK_N1, BLOCK_K1,
            )
        else:
            W13_e = _dequant_weight(gemm1_weights[le], gemm1_weights_scale[le], N_gemm1, H, BLOCK, device)
            A_e = A_dequant.index_select(0, token_idx)
            torch.mm(A_e, W13_e.t(), out=G1)
            del W13_e, A_e

        # ===== SwiGLU =====
        C = C_buf[:Tk]
        fused_silu_mul_kernel[(Tk, 1)](G1, C, Tk, I, 2048)

        # ===== GEMM2 + weighted scatter =====
        w_tok = local_weights.index_select(0, token_idx)[:, le].contiguous()

        if use_fused:
            BLOCK_N2 = 64
            BLOCK_K2 = 128
            num_m2 = (Tk + BLOCK_M - 1) // BLOCK_M
            num_n2 = H // BLOCK_N2
            fused_gemm2_weighted_scatter_kernel[(num_m2 * num_n2,)](
                C, gemm2_weights[le], gemm2_weights_scale[le],
                token_idx, w_tok, output,
                Tk, H, I, BLOCK, H_blocks, I_blocks,
                BLOCK_M, BLOCK_N2, BLOCK_K2,
            )
        else:
            W2_e = _dequant_weight(gemm2_weights[le], gemm2_weights_scale[le], H, I, BLOCK, device)
            O = O_buf[:Tk]
            torch.mm(C, W2_e.t(), out=O)
            del W2_e

            BH_ACC = 1024
            grid_acc = (Tk, (H + BH_ACC - 1) // BH_ACC)
            weighted_scatter_add_kernel[grid_acc](O, output, token_idx, w_tok, Tk, H, BH_ACC)

    result = output.to(torch.bfloat16)
    if orig_device.type != 'cuda':
        result = result.to(orig_device)
    return result
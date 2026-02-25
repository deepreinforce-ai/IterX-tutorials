import torch
import triton
import triton.language as tl

# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def _dequant_hidden_kernel(
    x_ptr, scale_ptr, out_ptr,
    T, H,
    BLOCK: tl.constexpr,
):
    """Dequant FP8 hidden states with block scales.
    x: [T, H] fp8, scale: [num_blocks, T] fp32 → out: [T, H] fp32
    """
    row = tl.program_id(0)   # token index
    blk = tl.program_id(1)   # block index

    # scale layout is [num_blocks, T], so scale[blk, row]
    scale = tl.load(scale_ptr + blk * T + row)

    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(x_ptr + row * H + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + row * H + offs, x * scale, mask=mask)


@triton.jit
def _swiglu_kernel(
    gate_ptr, up_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SwiGLU: out = silu(gate) * up, where silu(x) = x * sigmoid(x)"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask).to(tl.float32)

    result = (g * tl.sigmoid(g)) * u
    tl.store(out_ptr + offs, result, mask=mask)


# ============================================================================
# Helper Functions
# ============================================================================

def dequant_hidden(hidden_states, scale):
    """Dequant hidden states [T, H] fp8 with scale [num_blocks, T]."""
    T, H = hidden_states.shape
    BLOCK = 128
    num_blocks = H // BLOCK
    hidden_states = hidden_states.contiguous()
    scale = scale.contiguous()
    out = torch.empty(T, H, dtype=torch.float32, device=hidden_states.device)
    _dequant_hidden_kernel[(T, num_blocks)](
        hidden_states, scale, out, T, H, BLOCK=BLOCK,
    )
    return out


def dequant_weight_2d(weight, scale, BLOCK=128):
    """Dequant 2D weight [out_dim, in_dim] fp8 with scale [out_dim/B, in_dim/B]."""
    w = weight.to(torch.float32)
    s = scale.to(torch.float32)
    s = s.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return w * s


def swiglu(gate, up):
    """Fused SwiGLU using Triton kernel."""
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate, dtype=torch.float32)
    N = gate.numel()
    _swiglu_kernel[(triton.cdiv(N, 1024),)](gate, up, out, N, BLOCK_SIZE=1024)
    return out


# ============================================================================
# Entry Point
# ============================================================================

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
    # Constants (DeepSeek-V3/R1 geometry)
    H = 7168
    I = 2048
    BLOCK = 128
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    device = hidden_states.device

    # ---- 1) Dequant hidden states (Triton) ----
    A = dequant_hidden(hidden_states, hidden_states_scale)

    # ---- 2) Routing (PyTorch) ----
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = torch.sigmoid(logits)                                      # [T, E_global]
    s_with_bias = s + bias

    # Group scoring: top-2 per group → pick top-4 groups
    group_size = E_global // N_GROUP                               # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)        # [T, 8, 32]
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)                            # [T, 8]

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros(T, N_GROUP, device=device, dtype=torch.float32)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(2)
        .expand(T, N_GROUP, group_size)
        .reshape(T, E_global)
    )

    # Global top-k within kept groups
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    # Combination weights (use s without bias)
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor      # [T, E_global]

    # ---- 3) Expert computation ----
    output = torch.zeros(T, H, dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        # Find tokens that selected this expert
        sel_mask = (topk_idx == ge).any(dim=1)                     # [T] bool
        if not sel_mask.any():
            continue

        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)  # [Tk]
        A_e = A[token_idx]                                         # [Tk, H]

        # Lazy per-expert weight dequant (avoids materializing all weights upfront)
        W13_e = dequant_weight_2d(
            gemm1_weights[le], gemm1_weights_scale[le], BLOCK,
        )                                                          # [2I, H]
        W2_e = dequant_weight_2d(
            gemm2_weights[le], gemm2_weights_scale[le], BLOCK,
        )                                                          # [H, I]

        # GEMM1: [Tk, H] @ [H, 2I] → [Tk, 2I]
        G1 = A_e @ W13_e.t()

        # Fused SwiGLU (Triton): silu(gate) * up
        C = swiglu(G1[:, I:], G1[:, :I])                          # [Tk, I]

        # GEMM2: [Tk, I] @ [I, H] → [Tk, H]
        O = C @ W2_e.t()

        # Weighted accumulation
        w_tok = weights[token_idx, ge].unsqueeze(1)                # [Tk, 1]
        output.index_add_(0, token_idx, O * w_tok)

    return output.to(torch.bfloat16)

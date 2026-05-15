import torch
import torch.nn.functional as F


def _route_noaux_topk8(routing_logits: torch.Tensor,
                       routing_bias: torch.Tensor,
                       routed_scaling_factor: float):
    """Exact DeepSeek-style no-aux routing."""
    logits = routing_logits.float()
    bias = routing_bias.float().view(1, -1)

    s = torch.sigmoid(logits)
    s_bias = s + bias

    T, E = s.shape
    N_GROUP = 8
    GROUP_SIZE = E // N_GROUP
    TOPK_GROUP = 4
    TOP_K = 8

    sbg = s_bias.view(T, N_GROUP, GROUP_SIZE)
    top2 = torch.topk(sbg, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2.sum(dim=2)
    top_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=True).indices

    group_mask = torch.zeros((T, N_GROUP), device=logits.device, dtype=torch.bool)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E)

    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=torch.float32)
    pruned = torch.where(expert_mask, s_bias, neg_inf)
    topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=True).indices

    chosen = torch.zeros_like(s, dtype=torch.bool)
    chosen.scatter_(1, topk_idx, True)
    weights = torch.where(chosen, s, torch.zeros_like(s))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor
    topk_weights = torch.gather(weights, 1, topk_idx)
    return topk_idx, topk_weights


def _build_local_dispatch(topk_idx: torch.Tensor,
                          topk_weights: torch.Tensor,
                          local_expert_offset: int,
                          e_local: int):
    """Build GPU dispatch structures."""
    local_start = int(local_expert_offset)
    mask = (topk_idx >= local_start) & (topk_idx < local_start + e_local)
    tok_idx, slot_idx = torch.nonzero(mask, as_tuple=True)
    if tok_idx.numel() == 0:
        return None

    local_expert_idx = (topk_idx[tok_idx, slot_idx] - local_start).to(torch.int32)
    rw = topk_weights[tok_idx, slot_idx].to(torch.float32)

    perm = torch.argsort(local_expert_idx)
    sorted_tokens = tok_idx[perm].to(torch.int32)
    sorted_experts = local_expert_idx[perm]
    sorted_weights = rw[perm]

    counts = torch.bincount(sorted_experts.to(torch.int64), minlength=e_local)
    offsets = torch.zeros(e_local + 1, device=topk_idx.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts.to(torch.int32), dim=0)

    return sorted_tokens, sorted_experts, sorted_weights, offsets


def _dequant_hidden_states_blockwise(hidden_states: torch.Tensor, 
                                     hidden_states_scale: torch.Tensor,
                                     token_ids: torch.Tensor) -> torch.Tensor:
    """Dequantize hidden states with exact block-scale semantics."""
    # hidden_states: [T, H] fp8
    # hidden_states_scale: [H//128, T] fp32
    # token_ids: [Tk] int32
    # Returns: [Tk, H] fp32
    
    H = hidden_states.shape[1]
    Tk = token_ids.shape[0]
    
    # Gather tokens
    hs_gathered = hidden_states[token_ids.long()].float()  # [Tk, H]
    
    # Reshape for block-wise scaling
    hs_blocks = hs_gathered.view(Tk, H // 128, 128)  # [Tk, Hb, 128]
    
    # Gather scales: [Hb, Tk]
    scales_gathered = hidden_states_scale[:, token_ids.long()]  # [Hb, Tk]
    scales_gathered = scales_gathered.permute(1, 0).unsqueeze(-1)  # [Tk, Hb, 1]
    
    # Apply block scales
    hs_dequant = hs_blocks * scales_gathered  # [Tk, Hb, 128]
    hs_dequant = hs_dequant.reshape(Tk, H)  # [Tk, H]
    
    return hs_dequant


def _dequant_gemm1_weight_expert(w_fp8_e: torch.Tensor, 
                                 w_scale_e: torch.Tensor) -> torch.Tensor:
    """Dequantize GEMM1 weight for one expert with exact block-scale semantics."""
    # w_fp8_e: [2I, H] fp8
    # w_scale_e: [2I//128, H//128] fp32
    # Returns: [2I, H] fp32
    
    N, H = w_fp8_e.shape
    w = w_fp8_e.float().view(N // 128, 128, H // 128, 128)  # [Nb, 128, Hb, 128]
    w = w.permute(0, 2, 1, 3)  # [Nb, Hb, 128, 128]
    w = w * w_scale_e[:, :, None, None]  # Apply block scales
    w = w.permute(0, 2, 1, 3).reshape(N, H)  # [2I, H]
    return w


def _dequant_gemm2_weight_expert(w_fp8_e: torch.Tensor, 
                                 w_scale_e: torch.Tensor) -> torch.Tensor:
    """Dequantize GEMM2 weight for one expert with exact block-scale semantics."""
    # w_fp8_e: [H, I] fp8
    # w_scale_e: [H//128, I//128] fp32
    # Returns: [H, I] fp32
    
    H, I = w_fp8_e.shape
    w = w_fp8_e.float().view(H // 128, 128, I // 128, 128)  # [Hb, 128, Ib, 128]
    w = w.permute(0, 2, 1, 3)  # [Hb, Ib, 128, 128]
    w = w * w_scale_e[:, :, None, None]  # Apply block scales
    w = w.permute(0, 2, 1, 3).reshape(H, I)  # [H, I]
    return w


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,        # [T, 256] float32
    routing_bias: torch.Tensor,          # [256]    bfloat16
    hidden_states: torch.Tensor,         # [T, 7168] fp8
    hidden_states_scale: torch.Tensor,   # [56, T]  float32
    gemm1_weights: torch.Tensor,         # [32, 4096, 7168] fp8
    gemm1_weights_scale: torch.Tensor,   # [32, 32, 56] float32
    gemm2_weights: torch.Tensor,         # [32, 7168, 2048] fp8
    gemm2_weights_scale: torch.Tensor,   # [32, 56, 16] float32
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    device = hidden_states.device
    T = routing_logits.shape[0]
    H = 7168
    I = 2048
    E_LOCAL = gemm1_weights.shape[0]

    # 1) Exact routing
    topk_idx, topk_weights = _route_noaux_topk8(
        routing_logits, routing_bias, routed_scaling_factor
    )

    # 2) GPU dispatch
    dispatch = _build_local_dispatch(topk_idx, topk_weights, local_expert_offset, E_LOCAL)
    if dispatch is None:
        return torch.zeros((T, H), device=device, dtype=torch.bfloat16)

    sorted_tokens, sorted_experts, sorted_weights, expert_offsets = dispatch
    tk_total = int(sorted_tokens.numel())
    if tk_total == 0:
        return torch.zeros((T, H), device=device, dtype=torch.bfloat16)

    # 3) Process each expert with exact block-scale dequantization
    grouped_out = torch.zeros((tk_total, H), device=device, dtype=torch.float32)

    # Cache dequantized weights for active experts
    active_experts = torch.nonzero((expert_offsets[1:] - expert_offsets[:-1]) > 0, as_tuple=False).flatten()
    
    for e_t in active_experts.tolist():
        start = int(expert_offsets[e_t].item())
        end = int(expert_offsets[e_t + 1].item())
        if end <= start:
            continue

        # Get token indices for this expert
        expert_tokens = sorted_tokens[start:end]
        
        # Dequantize hidden states for these tokens
        hs_dequant = _dequant_hidden_states_blockwise(
            hidden_states, hidden_states_scale, expert_tokens
        )  # [tk_e, H] fp32
        
        # Dequantize GEMM1 weights for this expert
        w1_dequant = _dequant_gemm1_weight_expert(
            gemm1_weights[e_t], gemm1_weights_scale[e_t]
        )  # [2I, H] fp32
        
        # GEMM1: [tk_e, H] @ [H, 2I] -> [tk_e, 2I]
        g1 = torch.mm(hs_dequant, w1_dequant.t())  # FP32 accumulation
        
        # SwiGLU in FP32
        x1 = g1[:, :I]
        x2 = g1[:, I:]
        s = F.silu(x2) * x1  # [tk_e, I] fp32
        
        # Dequantize GEMM2 weights for this expert
        w2_dequant = _dequant_gemm2_weight_expert(
            gemm2_weights[e_t], gemm2_weights_scale[e_t]
        )  # [H, I] fp32
        
        # GEMM2: [tk_e, I] @ [I, H] -> [tk_e, H]
        y = torch.mm(s, w2_dequant.t())  # FP32 accumulation
        
        # Apply routing weights
        y = y * sorted_weights[start:end, None]
        
        # Store in grouped output
        grouped_out[start:end] = y

    # 4) Scatter-add to final output
    output = torch.zeros((T, H), device=device, dtype=torch.float32)
    output.index_add_(0, sorted_tokens.to(torch.int64), grouped_out)
    
    return output.to(torch.bfloat16)
import torch
import torch.nn.functional as F

# Constants for DeepSeek-V3 / R1 local MoE topology
_H = 7168
_I = 2048
_BLOCK = 128
_N_GROUP = 8
_TOPK_GROUP = 4
_TOP_K = 8


def _expand_scales_2d(scale_2d: torch.Tensor, rows_repeat: int, cols_repeat: int) -> torch.Tensor:
    # scale_2d: [Rb, Cb] -> [Rb*rows_repeat, Cb*cols_repeat]
    return scale_2d.repeat_interleave(rows_repeat, dim=0).repeat_interleave(cols_repeat, dim=1)


def _dequant_hidden_rows(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    # hidden_states: [T, H] fp8
    # hidden_states_scale: [H//128, T]
    # token_ids: [Tk]
    a_fp32 = hidden_states.index_select(0, token_ids).to(torch.float32)
    s = hidden_states_scale.index_select(1, token_ids).transpose(0, 1).contiguous()  # [Tk, 56]
    s = s.repeat_interleave(_BLOCK, dim=1)  # [Tk, H]
    return a_fp32 * s


def _dequant_gemm1_weight_full(
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    # w_fp8: [4096, 7168], w_scale: [32, 56]
    s = _expand_scales_2d(w_scale, _BLOCK, _BLOCK)
    return w_fp8.to(torch.float32) * s


def _dequant_gemm2_weight_full(
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    # w_fp8: [7168, 2048], w_scale: [56, 16]
    s = _expand_scales_2d(w_scale, _BLOCK, _BLOCK)
    return w_fp8.to(torch.float32) * s


def _run_expert_safe(
    token_ids: torch.Tensor,
    routing_w: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_w_e: torch.Tensor,
    gemm1_s_e: torch.Tensor,
    gemm2_w_e: torch.Tensor,
    gemm2_s_e: torch.Tensor,
) -> torch.Tensor:
    # Exact per-expert path in FP32.
    a = _dequant_hidden_rows(hidden_states, hidden_states_scale, token_ids)  # [Tk, H]
    w1 = _dequant_gemm1_weight_full(gemm1_w_e, gemm1_s_e)  # [2I, H]
    g1 = a @ w1.t()  # [Tk, 2I], fp32

    x1 = g1[:, :_I]
    x2 = g1[:, _I:]
    s = F.silu(x2) * x1  # [Tk, I], fp32

    w2 = _dequant_gemm2_weight_full(gemm2_w_e, gemm2_s_e)  # [H, I]
    y = s @ w2.t()  # [Tk, H], fp32
    y.mul_(routing_w.unsqueeze(1))
    return y


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
    E_global = routing_logits.shape[1]
    E_local = gemm1_weights.shape[0]

    # 1) Exact DeepSeek no-aux routing
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).view(1, E_global)

    s = torch.sigmoid(logits)                    # [T, 256]
    s_bias = s + bias                           # [T, 256]

    group_size = E_global // _N_GROUP           # 32
    grouped = s_bias.view(T, _N_GROUP, group_size)
    top2_vals = torch.topk(grouped, k=2, dim=2, largest=True, sorted=True).values
    group_scores = top2_vals.sum(dim=2)         # [T, 8]

    top_groups = torch.topk(group_scores, k=_TOPK_GROUP, dim=1, largest=True, sorted=True).indices
    group_mask = torch.zeros((T, _N_GROUP), device=device, dtype=torch.bool)
    group_mask.scatter_(1, top_groups, True)
    expert_mask = group_mask.unsqueeze(2).expand(T, _N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
    pruned_scores = torch.where(expert_mask, s_bias, neg_inf)
    topk_idx = torch.topk(pruned_scores, k=_TOP_K, dim=1, largest=True, sorted=True).indices  # [T, 8]

    topk_sigmoid = s.gather(1, topk_idx)  # weights from original sigmoid(logits)
    denom = topk_sigmoid.sum(dim=1, keepdim=True) + 1e-20
    topk_weights = topk_sigmoid / denom
    topk_weights.mul_(float(routed_scaling_factor))

    # 2) Build local dispatch on GPU
    local_start = int(local_expert_offset)
    local_end = local_start + E_local

    local_mask = (topk_idx >= local_start) & (topk_idx < local_end)
    flat_token_idx, flat_slot_idx = torch.nonzero(local_mask, as_tuple=True)

    if flat_token_idx.numel() == 0:
        return torch.zeros((T, _H), dtype=torch.bfloat16, device=device)

    local_expert_idx = (topk_idx[flat_token_idx, flat_slot_idx] - local_start).to(torch.int64)
    local_weights = topk_weights[flat_token_idx, flat_slot_idx].to(torch.float32)

    order = torch.argsort(local_expert_idx)
    sorted_tokens = flat_token_idx[order].to(torch.int64)
    sorted_experts = local_expert_idx[order]
    sorted_weights = local_weights[order]

    expert_counts = torch.bincount(sorted_experts, minlength=E_local)
    expert_offsets = torch.zeros(E_local + 1, device=device, dtype=torch.int64)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    # 3) Per-expert exact execution
    output = torch.zeros((T, _H), dtype=torch.float32, device=device)

    for e in range(E_local):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if start == end:
            continue

        tok_e = sorted_tokens[start:end]
        w_e = sorted_weights[start:end]

        y_e = _run_expert_safe(
            tok_e,
            w_e,
            hidden_states,
            hidden_states_scale,
            gemm1_weights[e],
            gemm1_weights_scale[e],
            gemm2_weights[e],
            gemm2_weights_scale[e],
        )
        output.index_add_(0, tok_e, y_e)

    return output.to(torch.bfloat16)